# HAWFormer

Hierarchy-Aware Adaptive Graph Learning and Band-Wise Wavelet Transformer for
traffic flow prediction.

HAWFormer has two modules. **HAGL** infers the dependency graph from three
scales — macroscopic node attributes, microscopic trajectory semantics, and a
mesoscopic regional organization re-estimated from the graph the model has
learned — and fuses them with an entry-wise gate. **BWST** predicts from that
graph: the encoded sequence is split into Haar bands before any spatial operator
runs, each band gets its own dynamic relation mask and neighborhood budget, and
the full horizon is decoded in one pass from a persistence-anchored head. The
two modules are trained in alternation.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. Training used a single NVIDIA RTX 4060 Laptop GPU; the code runs
on CPU but the 250-epoch schedule is impractical there.

`pymetis` is optional. Without it, `hawformer/models/hierarchy.py` falls back to
a spectral partitioner with greedy rebalancing — a different algorithm with
different balance guarantees, so install it if you want the multilevel scheme.

## Data

Neither dataset is redistributed here.

**Qingdao** — node-level passage counts plus taxi trajectories, 1–19 August
2019, 06:00–18:00 daily. 134 nodes, 2736 five-minute steps.

```bash
python3 scripts/prepare_qdb.py --raw /path/to/qingdao
```

**T-Drive** — 10,357 Beijing taxis, 2–8 February 2008
([Kaggle](https://www.kaggle.com/datasets/arashnic/tdriver)). Nodes are the 150
busiest GeoHash cells at precision 6; flow is the number of *distinct* taxis per
five-minute bin, not the raw GPS point count.

```bash
python3 scripts/prepare_data.py --config configs/tdrive.yaml
```

Then set the three `raw:` paths in `configs/tdrive.yaml` to where the prepared
files landed.

## Reproduce

```bash
./run_qingdao.sh      # three seeds, complete model
./run_tdrive.sh       # three seeds, complete model
./run_baselines.sh    # the nine competitors, same partitions, same budget
./run_ablations.sh    # component ablations
```

Or a single run:

```bash
python3 scripts/train.py --config configs/qdb19.yaml --tag final --seed 0
```

Any setting can be overridden from the command line:

```bash
python3 scripts/train.py --config configs/qdb19.yaml --set model.d_model=32 --set train.epochs=60
```

## Verify without retraining

Every reported number comes from a stored prediction tensor re-scored under one
metric implementation. `results/` ships those tensors, so the table can be
checked directly:

```bash
python3 scripts/verify_results.py
```

```
qdb19
  seed0        MAE   8.5129   MAPE  0.0933   RMSE  13.7517
  seed1        MAE   8.5988   MAPE  0.0941   RMSE  13.8006
  seed2        MAE   8.5979   MAPE  0.0926   RMSE  13.8471
  mean +/- sd  MAE   8.5699 +/- 0.0493   MAPE  0.0933   RMSE  13.7998

tdrive
  seed0        MAE   1.1630   MAPE  0.1787   RMSE   1.7476
  seed1        MAE   1.1703   MAPE  0.1628   RMSE   1.7666
  seed2        MAE   1.1670   MAPE  0.1650   RMSE   1.7547
  mean +/- sd  MAE   1.1668 +/- 0.0036   MAPE  0.1688   RMSE   1.7563
```

Point it at your own runs with `--runs runs`.

## Configuration

`configs/default.yaml` holds the shared settings; the dataset configs inherit
and override. The values shipped here are the ones the runs in `results/` used —
`results/*/seed0/report.json` records each run's fully-resolved configuration.

| | |
|---|---|
| History / horizon | `T_h = T_p = 12` (one hour in, one hour out at 5 min) |
| Hidden width | `d = 64`, 2 BWST encoder blocks, 1 decoder |
| Attention heads | 4 on Qingdao, 2 on T-Drive; `M = 2` similarity heads |
| Wavelet | `J = 2` Haar levels → 3 bands |
| Band budget | `κ_j = max(κ_min, ⌈κ₀γ^j⌉)`, `κ₀=32 γ=0.5 κ_min=4` → `[32, 16, 8]` |
| Graph | `k_geo = 12` structural support, top-20 row sparsity, Chebyshev `K = 2` |
| Hierarchy | `Q = 8` structural groups, `P` by silhouette, refresh every `R = 3` |
| Graph bank | size 3, loss-derived fusion weights |
| Optimizer | AdamW, `lr = 3e-4`, weight decay `1e-4`, Huber `δ = 1.0` |
| Regularization | `λ_a = 1e-5` on the fused graph, `λ_c = 1e-4` on affinities |
| Budget | 250 epochs, patience 30 — the same budget given to every baseline |

Band index `j = 0` is the approximation band (lowest frequencies); `j = 1…J` are
the detail bands ordered from the finest scale outward. The budget contracts
across the hierarchy from the approximation outward.

## Layout

```
hawformer/
  data/      dataset assembly, GeoHash tokenization, Kalman smoothing,
             microscopic (Word2Vec) graph construction
  models/    hagl.py, bwst.py, wavelet.py, hierarchy.py, hawformer.py
  train.py   alternating loop, graph bank, hierarchy refresh
  metrics.py single scoring implementation used for every model
scripts/     training, data preparation, baselines, ablations, tables, figures
configs/     default + one per dataset
results/     stored predictions, reports and learned structures for the runs above
```

## Known discrepancy

The numbers above are what this code produces. They differ slightly from the
values printed in the manuscript (Qingdao MAE 8.499, T-Drive 1.150), which were
taken from a different set of runs. Resolve this before release: either re-derive
the table from `results/`, or identify and document the configuration the
manuscript's numbers came from.

Two smaller items in the same category:

- `train.loss_space` is set to `original` here because that is what the reported
  runs used. The manuscript states the loss is taken on normalized targets. One
  of the two needs to change.
- `hawformer/data/kalman.py` is a forward–backward (RTS) smoother applied in
  `dataset.py` before the split indices are computed, so each estimate sees the
  whole series. For a strictly causal protocol, smooth each partition separately
  after splitting.

## License

MIT — see [LICENSE](LICENSE). The datasets are not covered by it and remain
under their own terms.

## Citation

```bibtex
@article{arshad2026hawformer,
  title   = {HAWFormer: Hierarchy-Aware Adaptive Graph Learning and Band-Wise
             Wavelet Transformer for Traffic Flow Prediction},
  author  = {Arshad, Muhammad Usman and Zhou, Kuanjiu and Li, Yicong},
  year    = {2026}
}
```
