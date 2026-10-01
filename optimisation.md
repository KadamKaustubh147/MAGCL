# Jaccard graph construction: dense → sparse

## Where

`_jaccard_side()` in [src/magcl/graphs.py](src/magcl/graphs.py), used by `high_order_adj()` to
build the high-order user-user (P) and item-item (Q) graphs (Eqs 2-3 of the paper).

## The problem

The function computes, for every node `x`, its Jaccard similarity to every other
node `y` that shares at least one neighbour, then keeps `y` if `JSC(x,y) >= beta`
or `y` is among `x`'s top-S most similar nodes (Eq 3).

The original implementation computed this by **densifying** `R @ R.T`:

```python
chunk = max(1, int(2e7 // max(k, 1)))          # cap each dense block at ~20M entries
for lo in range(0, k, chunk):
    hi = min(lo + chunk, k)
    inter = (R[lo:hi] @ R.T).toarray().astype(np.float64)   # dense block
    ...
```

`k` is the number of nodes on that side (users for P, items for Q). The `chunk`
size bounds how much memory any *one* block uses (~20M float64 entries ≈ 160 MB),
but it does nothing about *total work*: however you chunk it, materialising
`R @ R.T` densely means computing and writing out **k² entries** in total, most
of which (`inter[x,y] == 0`, i.e. no shared neighbour) are immediately discarded
by the very next line (`valid = inter > 0`).

- **ML-1M**: k = 6,040 users / 3,629 items → k² is small, finishes in seconds.
- **Alibaba-iFashion**: k = 300,001 users / 81,615 items → **~2,460x** more work
  on the user side and **~505x** more on the item side than ML-1M. In practice
  this made graph construction run for 45+ minutes with no sign of finishing,
  before also crashing later with `CUDA error: out of memory` when the
  (needlessly huge) resulting tensors were moved to the GPU.

This was a pure **implementation** bottleneck, not a data or memory problem —
confirmed by watching system RAM stay flat (not climbing toward exhaustion)
while the process ran for 45 minutes without completing.

## The fix

Compute `R @ R.T` as a **sparse × sparse** product and never call `.toarray()`.
scipy's sparse matmul only produces a nonzero entry at `(x, y)` where rows `x`
and `y` of `R` already share a column (a neighbour) — which is *exactly* the
`inter > 0` condition the old code was filtering down to anyway. So working
directly on the sparse result doesn't change which pairs are considered; it
just skips ever computing/writing the entries that were always going to be
thrown away.

```python
inter = (R @ R.T).tocsr()                   # nonzero only where x,y share a neighbour
inter = inter - sp.diags(inter.diagonal())  # drop x == y
inter.eliminate_zeros()

for x in range(k):
    start, end = indptr[x], indptr[x + 1]
    if start == end:
        continue
    cols_x = indices[start:end]
    inter_x = data[start:end]
    jsc_x = inter_x / np.maximum(deg[x] + deg[cols_x] - inter_x, 1e-12)

    if len(jsc_x) <= top_s:
        keep_x = np.ones(len(jsc_x), dtype=bool)     # fewer neighbours than S: keep all
    else:
        kth = np.partition(jsc_x, -top_s)[-top_s]
        keep_x = (jsc_x >= beta) | (jsc_x >= kth)
    ...
```

Now the work scales with **actual co-occurrence density** (`nnz(R @ R.T)`),
not with `k²`. For real interaction data this is far smaller than the full
dense grid, since most pairs of users/items never share a neighbour at all.

## Why it's not an approximation

Every value computed is identical to before:

- Eq 2's intersection count `|N(x) ∩ N(y)|` is the same `R @ R.T` entry,
  sparse or dense — sparsity just decides which entries get *materialised*.
- The old code masked invalid pairs (no shared neighbour, or `x == y`) to
  `-1` before taking each row's S-th-largest value, specifically so they
  could never be selected (a real JSC score is always ≥ 0). The new code
  just never creates those `-1` placeholders — the true S-th-largest value
  among the *valid* candidates is unchanged either way.
- Self-pairs are excluded in both versions.

**Verified, not just argued**: ran both the old dense implementation and the
new sparse one side-by-side on the real ML-1M interaction matrix (both the
user-user P side and item-item Q side). Result: bit-for-bit identical edge
sets — 30,700 edges on the P side, 18,912 on the Q side, zero differences.

## Result

| | Old (dense, chunked) | New (sparse) |
|---|---|---|
| ML-1M (k ≈ 6k) | seconds | seconds (unchanged) |
| Alibaba-iFashion (k up to 300k) | 45+ min, never finished; then crashed with CUDA OOM | **~63s** graph construction, **~24s** including GPU transfer |

GPU memory after the fix: 421.6 MB (previously OOM'd on a 6 GB card, because
the tensors being moved to GPU were far larger than they needed to be).

---

# `full_sort_predict` recomputing `forward()` every eval batch

## Where

`full_sort_predict()` in [src/magcl/model.py](src/magcl/model.py).

## The problem

RecBole calls `full_sort_predict()` once per evaluation batch, but `forward()`
-- the whole L-layer graph propagation over all three graphs -- doesn't depend
on which users are in that particular batch. The original code called
`self.forward()` unconditionally on every invocation, recomputing the entire
propagation from scratch every batch instead of once per evaluation pass.

## The fix

Cache the result across calls within one evaluation pass, invalidated inside
`calculate_loss()` (called every training step, since training changes `E0`).
Implemented via `other_parameter_name = ["restore_e"]`, exactly matching
RecBole's own official `LightGCN` (`restore_user_e`/`restore_item_e`) --
this is RecBole's own mechanism for persisting plain (non-`nn.Parameter`)
attributes through checkpoint save/load. That detail matters: without it, the
cache would silently survive a checkpoint reload for final test evaluation
inconsistent with the reloaded `E0` -- a correctness bug, not just a
performance one -- whenever early stopping doesn't trigger on the literal
last epoch.

## Result

Measured on ML-1M: full-sort eval **103s -> 5.7s** (18x). Verified the smoke
test still produces finite losses/gradients afterward.

---

# `eval_batch_size` silently flooring to 1 user per eval step

## Where

Not our code at all -- a RecBole config value (`eval_batch_size` in every
`configs/*.yaml`) being misunderstood. The actual behaviour lives in
`FullSortEvalDataLoader._init_batch_size_and_step` (RecBole's own
`recbole/data/dataloader/general_dataloader.py`):

```python
batch_num = max(batch_size // self._dataset.item_num, 1)
```

## The problem

`eval_batch_size` is not "users per batch" -- it's a budget on total
score-matrix cells, and RecBole derives `users_per_step = eval_batch_size //
n_items` from it. Our configs used the common default `eval_batch_size:
4096`. For ML-1M (`n_items=3629`), `4096 // 3629 = 1`. For Alibaba-iFashion
(`n_items=81615`), `4096 // 81615 = 0`, floored to the minimum of `1`.
**Both datasets were silently evaluating exactly one user per eval step**,
regardless of the configured batch size.

This is invisible at ML-1M's scale: ~6,040 single-user steps is cheap, and
was already masked by the `full_sort_predict` caching fix above (103s ->
5.7s looked like a full fix, but a real bottleneck was still hiding
underneath, just small enough not to matter yet). At Alibaba-iFashion's
scale, ~300,001 single-user steps meant RecBole's own per-step result
accumulator (`Collector.eval_batch_collect` -> `DataStruct.update_tensor`,
which does `torch.cat((accumulated_so_far, new_batch))` -- a growing
re-copy -- **on every single step**) became quadratic in the number of
users. Confirmed directly: `len(valid_data)` (the real number of eval
iterations, not a batch count) was **271,235** at the default setting.
Measured full-sort eval time at that setting: **14,176.5s (3.94 hours)**,
even with the `full_sort_predict` caching fix already applied.

## The fix

Raise `eval_batch_size` well above `n_items` so multiple users batch
together per step:

- ML-1M: `4096000` (~1,128 users/step)
- Alibaba-iFashion: `40000000` (~490 users/step; verified via
  `valid_data.step` and `len(valid_data)` before committing to a full timed
  run)

Purely a config change -- no code touched, no equations changed, no
deviation from "stock RecBole."

## Why it's not an approximation

Batching users together for full-sort scoring doesn't change which items get
scored, masked, or ranked for any individual user -- it's the same
computation, just grouped. Verified: metrics before/after (recall@10/20/50,
ndcg@10/20/50) matched to within float-accumulation-order noise
(e.g. recall@20 0.0284 -> 0.0288 -- the kind of tiny non-determinism expected
from batched vs. per-row floating-point operations, not a sign of a
different computation).

## Result

| | eval_batch_size=4096 (default) | eval_batch_size=40000000 (fixed) |
|---|---|---|
| Eval steps | 271,235 | 554 |
| Full-sort eval time (Alibaba-iFashion) | 14,176.5s (3.94 hours) | **43.7s** |

**324x speedup**, on top of the 18x from the `full_sort_predict` caching fix.
Combined, this is why `eval_step`/`stopping_step` no longer need the
`5`/`2` tuning-speed approximation documented in `configs/*.yaml` -- both
configs now use the paper's literal `eval_step: 1` / `stopping_step: 10`
again, since full-sort eval is fast enough on its own merits.
