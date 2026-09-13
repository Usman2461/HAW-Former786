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
