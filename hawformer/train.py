"""Alternating optimisation: predictor, graph learner, graph bank, refresh.

The predictor consumes the learned graph, and the graph learner's mesoscopic
branch consumes a partition of that same graph.  Updating everything at once is
unstable, because the partition is a *discrete* function of a quantity that is
still moving: a small change in the graph can move a sensor between groups,
changing the pooled features the attention layers depend on.

So each epoch runs in stages:

  Stage 1  train the predictor with the graph learner frozen
  Stage 2  train the graph learner with the predictor frozen
  Stage 3  push the new graph into a fixed-size bank and fuse its members
           with weights set by the loss each induces, evicting the worst
  Stage 4  every R epochs, recompute the hierarchy from the fused graph

Stage 3 is what damps the variance introduced by the coupling: without it the
partition chases whatever graph the last few batches produced.
"""
from __future__ import annotations

import contextlib
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data.dataset import Bundle
from .metrics import all_metrics
from .models.hawformer import HAWFormer, huber_loss
from .models.hierarchy import build_hierarchy
from .utils import get_logger, save_json

LOG = get_logger()


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    lr_graph: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    huber_delta: float = 1.0
    loss_space: str = "original"
    lambda_a: float = 1e-5
    lambda_c: float = 1e-4
    bank_size: int = 3
    refresh_every: int = 5
    graph_steps: int = 20
    patience: int = 20
    warmup_epochs: int = 3
    alternating: bool = True
    use_bank: bool = True
    graph_every: int = 1        # run the graph stages every N epochs
    # "off" or "bf16".  bfloat16 keeps the fp32 exponent range, so no gradient
    # scaler is needed and the alternating optimisation cannot silently
    # underflow the way fp16 would; the loss is still taken in fp32.
    amp: str = "off"


def _autocast(device, amp: str):
    if amp == "bf16" and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class GraphBank:
    """Fixed-size bank of candidate graphs fused by inverse loss.

    Weights follow the softmax of ``max(L) - L_k`` so the best graph dominates
    while the others still contribute; the worst member is evicted each round
    to keep the bank size constant.
    """

    def __init__(self, size: int = 3):
        self.size = size
        self.graphs: List[torch.Tensor] = []
        self.losses: List[float] = []

    def add(self, A: torch.Tensor, loss: float) -> None:
        self.graphs.append(A.detach().clone())
        self.losses.append(float(loss))
        if len(self.graphs) > self.size:
            i = int(np.argmax(self.losses))
            self.graphs.pop(i)
            self.losses.pop(i)

    def fuse(self) -> torch.Tensor:
        if not self.graphs:
            raise RuntimeError("empty bank")
        l = np.asarray(self.losses)
        w = np.exp(l.max() - l)
        w = w / w.sum()
        out = torch.zeros_like(self.graphs[0])
        for wi, g in zip(w, self.graphs):
            out = out + float(wi) * g
        return out


def _to_device(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]


@torch.no_grad()
def evaluate(
    model: HAWFormer,
    loader,
    bundle: Bundle,
    device,
    mape_threshold: float = 5.0,
    return_arrays: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    model.eval()
    preds, trues, masks = [], [], []
    for batch in loader:
        x, y, m, ti, _ = _to_device(batch, device)
        out = model(x, ti)
        p = bundle.normalizer.inverse(out["pred"])
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
        masks.append(m.cpu().numpy())
    if not preds:
        return {k: float("nan") for k in
                ("MAE", "RMSE", "WAPE", "MAPE", "MAPE_coverage", "R2", "Bias")}, None
    P = np.concatenate(preds)
    Y = np.concatenate(trues)
    M = np.concatenate(masks)
    metrics = all_metrics(Y, P, M, mape_threshold)
    arrays = {"pred": P, "true": Y, "mask": M} if return_arrays else None
    return metrics, arrays


def _loss(pred_norm, pred_orig, y, m, bundle, tcfg):
    """Training loss, in the space `tcfg.loss_space` selects.

    The default -- Huber on the ORIGINAL scale with delta 1.0 -- is not the
    same objective on different datasets, which makes "identical settings
    across datasets" false in the place it matters most.  Typical errors are
    15.5 vehicles on Qingdao, 1.4 on T-Drive and 0.74 on Porto, so against a
    fixed delta of 1 the first trains under pure L1 and the last mostly under
    MSE.  The gradient reaching the output head is also multiplied by the
    inverse transform's std -- 143 on Qingdao against 2.1 on Porto -- so one
    learning rate means two things seventy-fold apart.

    Computing the loss on normalised targets makes delta a z-score (delta = 1
    is one standard deviation), which is the same objective everywhere and
    puts the datasets on one gradient scale.
    """
    if tcfg.loss_space == "normalized":
        t = (y - bundle.normalizer.mean) / bundle.normalizer.std
        return huber_loss(pred_norm, t, m, tcfg.huber_delta)
    return huber_loss(pred_orig, y, m, tcfg.huber_delta)


def _predictor_params(model: HAWFormer):
    hagl_ids = {id(p) for p in model.hagl.parameters()}
    return [p for p in model.parameters() if id(p) not in hagl_ids]


def train(
    model: HAWFormer,
    loaders: Dict[str, torch.utils.data.DataLoader],
    bundle: Bundle,
    tcfg: TrainConfig,
    device,
    out_dir: str,
    hierarchy_kwargs: Optional[dict] = None,
) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    opt_pred = torch.optim.AdamW(_predictor_params(model), lr=tcfg.lr,
                                 weight_decay=tcfg.weight_decay)
    opt_graph = torch.optim.AdamW(model.hagl.parameters(), lr=tcfg.lr_graph,
                                  weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_pred, mode="min", factor=0.5, patience=max(3, tcfg.patience // 4))

    bank = GraphBank(tcfg.bank_size)
    hk = hierarchy_kwargs or {}
    flow_train = bundle.flow[bundle.splits["train"][0]:bundle.splits["train"][1]]

    best = {"val_MAE": float("inf"), "epoch": -1}
    best_state = None
    history: List[Dict] = []
    bad = 0
    # Track the structural partition at every refresh.  The claim that the
    # hierarchy is *adaptive* is only checkable if we can see which sensors
    # moved once it was computed from the learned graph instead of from A.
    hier_history: List[np.ndarray] = [model.H_g.detach().cpu().numpy().argmax(1)]
    hier_epochs: List[int] = [0]

    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()

        # ---- Stage 1: predictor, graph frozen ----------------------------
        model.train()
        model.hagl.requires_grad_(False)
        tr_loss, nb = 0.0, 0
        for batch in loaders["train"]:
            x, y, m, ti, _ = _to_device(batch, device)
            with _autocast(device, tcfg.amp):
                out = model(x, ti)
            out["pred"] = out["pred"].float()
            pred = bundle.normalizer.inverse(out["pred"])
            loss = _loss(out["pred"], pred, y, m, bundle, tcfg)
            opt_pred.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(_predictor_params(model), tcfg.grad_clip)
            opt_pred.step()
            tr_loss += loss.detach().item()
            nb += 1
        model.hagl.requires_grad_(True)
        tr_loss = tr_loss / max(nb, 1)

        # ---- Stage 2: graph learner, predictor frozen --------------------
        # graph_every > 1 lets the graph move on a slower clock than the
        # predictor.  The ablations put the two best Qingdao variants at "never
        # refresh the hierarchy" and "never update the graph at all", which is
        # equally consistent with the representation being useless and with the
        # schedule perturbing the predictor faster than it can adapt; a slower
        # clock separates the two readings.
        graph_loss = float("nan")
        if (tcfg.alternating and epoch > tcfg.warmup_epochs
                and epoch % max(1, tcfg.graph_every) == 0):
            for p in _predictor_params(model):
                p.requires_grad_(False)
            gl, gn = 0.0, 0
            for i, batch in enumerate(loaders["train"]):
                if i >= tcfg.graph_steps:
                    break
                x, y, m, ti, _ = _to_device(batch, device)
                with _autocast(device, tcfg.amp):
                    g = model.learn_graph(x)
                    out = model(x, ti, A=g["A_sparse"])
                out["pred"] = out["pred"].float()
                g["A_star"] = g["A_star"].float()
                pred = bundle.normalizer.inverse(out["pred"])
                loss = _loss(out["pred"], pred, y, m, bundle, tcfg)
                loss = loss + model.hagl.regularisation(
                    g["A_star"], tcfg.lambda_a, tcfg.lambda_c)
                opt_graph.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.hagl.parameters(), tcfg.grad_clip)
                opt_graph.step()
                gl += loss.detach().item()
                gn += 1
            for p in _predictor_params(model):
                p.requires_grad_(True)
            graph_loss = gl / max(gn, 1)

            # ---- Stage 3: graph bank -------------------------------------
            with torch.no_grad():
                xb, yb, mb, tib, _ = _to_device(next(iter(loaders["train"])), device)
                g = model.learn_graph(xb)
                A_new = g["A_sparse"]
                if tcfg.use_bank:
                    out = model(xb, tib, A=A_new)
                    pr = bundle.normalizer.inverse(out["pred"])
                    bank.add(A_new, huber_loss(pr, yb, mb, tcfg.huber_delta).item())
                    A_new = bank.fuse()
                model.freeze_graph(A_new)

            # ---- Stage 4: slow hierarchy clock ---------------------------
            if (not model.cfg.static_hierarchy) and epoch % tcfg.refresh_every == 0:
                h = build_hierarchy(
                    flow_train, model.A_frozen.detach().cpu().numpy(),
                    P=model.cfg.P, Q=model.cfg.Q,
                    wavelet_levels=model.cfg.wavelet_levels,
                    use_spectral_descriptor=model.cfg.use_spectral_descriptor,
                    soft_membership=model.cfg.soft_membership,
                    topk=model.cfg.topk,
                    membership_temperature=model.cfg.membership_temperature,
                    reg_covar=model.cfg.reg_covar, **hk,
                )
                model.set_hierarchy(h.H_s, h.H_g)
                hier_history.append(h.H_g.argmax(1))
                hier_epochs.append(epoch)
                opt_graph = torch.optim.AdamW(model.hagl.parameters(),
                                              lr=tcfg.lr_graph,
                                              weight_decay=tcfg.weight_decay)

        # ---- validation ---------------------------------------------------
        val, _ = evaluate(model, loaders["val"], bundle, device)
        sched.step(val["MAE"])
        rec = {"epoch": epoch, "train_loss": tr_loss, "graph_loss": graph_loss,
               "secs": time.time() - t0,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        gtxt = "  --  " if np.isnan(graph_loss) else f"{graph_loss:.4f}"
        # Peak reserved, not allocated: on an 8 GB card the number that decides
        # whether the run stays resident or starts paging to host memory is what
        # the caching allocator holds, and epoch time creeping upward while this
        # climbs is the signature of that spill.
        mem = ""
        if getattr(device, "type", str(device)) == "cuda":
            mem = (f" | {torch.cuda.max_memory_reserved()/2**30:.2f}GB")
            rec["peak_gpu_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
        LOG.info("epoch %3d | train %.4f | graph %s | val MAE %.4f RMSE %.4f "
                 "WAPE %.2f%% | %.1fs%s",
                 epoch, tr_loss, gtxt, val["MAE"], val["RMSE"],
                 val["WAPE"], rec["secs"], mem)

        if val["MAE"] < best["val_MAE"] - 1e-6:
            best = {"val_MAE": val["MAE"], "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= tcfg.patience:
                LOG.info("early stop at epoch %d (best %d)", epoch, best["epoch"])
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_json(history, str(out_dir / "history.json"))

    # Persist the learned structures for the interpretability figures.  H_s and
    # H_g are non-persistent buffers, so without this they would be lost with
    # the process and the partition-migration figure could not be drawn.
    with torch.no_grad():
        omega = [torch.softmax(b.band_logits.detach(), 0).cpu().numpy()
                 for b in model.blocks]
        np.savez_compressed(
            out_dir / "structures.npz",
            A_learned=model.A_frozen.detach().cpu().numpy(),
            A_pre=model.A_pre.detach().cpu().numpy(),
            A_mi=model.A_mi.detach().cpu().numpy(),
            H_s=model.H_s.detach().cpu().numpy(),
            H_g=model.H_g.detach().cpu().numpy(),
            hier_history=np.stack(hier_history) if hier_history else np.zeros((0, 0)),
            hier_epochs=np.asarray(hier_epochs),
            band_weights=np.stack(omega) if omega else np.zeros((0, 0)),
            budgets=np.asarray(model.budgets),
            lambda_g=(torch.nn.functional.softplus(model.hagl.meso.L_g).detach()
                      .cpu().numpy() if model.hagl.meso is not None
                      else np.zeros((0, 0))),
            lambda_s=(torch.nn.functional.softplus(model.hagl.meso.L_s).detach()
                      .cpu().numpy() if model.hagl.meso is not None
                      else np.zeros((0, 0))),
        )
    return {"best": best, "history": history}
