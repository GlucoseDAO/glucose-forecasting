"""What does a pretrained JEPA encoder actually organise its latent space by?

The per-epoch plots in `jepa_pretrain.py` answer "is it collapsing?". This
answers the next question: given a *finished* encoder, does the embedding
separate by **dataset** (Loop vs AI-READI), by **patient**, or by **glucose
level** — and how does that change with the pretraining window (288 / 864 /
2016 steps = 1 / 3 / 7 days)?

Same recipe as panel 3 of the training diagnostics — encode a fixed sample of
windows, mean-pool the patch axis, PCA to 2D — but the scatter is drawn once per
colour variable over the *same* projection, so the panels are directly
comparable: whichever colour lines up with the geometry is what the encoder
encodes.

A fourth panel colours by window *shape* (`window_trend`, the pretraining
colouring) as the control. Without it there is no reference for "the encoder
does encode something", and a featureless dataset/patient/glucose panel cannot
be told apart from a dead encoder.

Read the glucose panel with the instance-norm caveat in mind: `JepaEncoder`
z-scores every window inside its own forward pass, so absolute level is removed
*by construction* and a null result there is the expected outcome, not a
finding. The `glucose_r2` column in `probe_metrics.csv` is what makes that
quantitative rather than a squint at a colourbar.

Windows come from ONE split (default: val), so the numbers describe the
encoder's behaviour on series it did not see — pretraining used train only.

    uv run python src/sugar_jepa/encoder_pca_probe.py --help
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from common.data.columns import (
    COL_EVENT,
    COL_GLU,
    COL_SEQ,
    COL_SPLIT,
    COL_TS_SHORT as COL_TS,
    COL_USER,
    TS_FORMAT,
)
from common.data.loading import impute_and_sort
from sugar_jepa.encoder_plots import window_trend
from sugar_jepa.sugar_jepa_model import JepaEncoder
from sugar_one.console_log import echo_plain

app = typer.Typer(
    name="encoder_pca_probe",
    add_completion=False,
    help="PCA of a pretrained JEPA encoder's latents, coloured by dataset / patient / glucose.",
)

DEFAULT_ENCODERS = (
    "data/output/runs/jepa_encoder-288",
    "data/output/runs/jepa_encoder-864",
    "data/output/runs/jepa_encoder-2016",
)

# Both id columns carry the provenance; sequence_id is the series key the
# encoders were pretrained on, so it is the one we key windows by.
DATASET_BY_PATIENT_PREFIX = {"loop": "loop", "ai_ready": "aiready"}
DATASET_BY_SEQ_PREFIX = {"L": "loop", "A": "aiready"}
DATASET_COLORS = {"loop": "#C44E52", "aiready": "#4C72B0", "unknown": "#999999"}


# ============================================================================
#  ENCODER RESOLUTION — a run dir, or a bare state_dict
# ============================================================================

@dataclass
class EncoderSpec:
    label: str
    path: Path
    window: int
    patch_size: int
    embed_dim: int
    n_layers: int
    n_heads: int
    norm: str


def _shape_from_state(state: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    """(window, patch_size, embed_dim, n_layers) read off the tensors themselves.

    Everything except head count is recoverable: patch_embed is Conv1d(1, E, k=P,
    stride=P) and pos_enc.pe is (1, n_patches, E). n_heads is folded into a single
    in_proj weight and cannot be recovered — it comes from config.json or a flag.
    """
    embed_dim, _, patch_size = state["patch_embed.weight"].shape
    n_patches = state["pos_enc.pe"].shape[1]
    n_layers = 1 + max(
        int(k.split(".")[1]) for k in state if k.startswith("blocks.")
    )
    return n_patches * patch_size, patch_size, embed_dim, n_layers


def resolve_encoder(spec: str, prefer: str, default_heads: int) -> EncoderSpec:
    """`spec` is a run directory, a parent holding latest.txt, or an encoder .pt."""
    p = Path(spec)
    cfg: dict = {}

    if p.is_dir():
        latest = p / "latest.txt"
        if not (p / "config.json").exists() and latest.exists():
            p = Path(latest.read_text().strip())
        if not (p / "config.json").exists():
            runs = sorted(d for d in p.iterdir() if d.is_dir() and (d / "config.json").exists())
            if not runs:
                raise typer.BadParameter(f"No pretraining run with a config.json under {spec}")
            p = runs[-1]
        candidates = [p / f"encoder_{prefer}.pt", p / "encoder.pt", p / "encoder_best.pt"]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            raise typer.BadParameter(f"No encoder*.pt in {p}")
        cfg = json.loads((p / "config.json").read_text())
    else:
        ckpt = p
        if not ckpt.exists():
            raise typer.BadParameter(f"No such encoder checkpoint: {ckpt}")
        sidecar = ckpt.parent / "config.json"
        if sidecar.exists():
            cfg = json.loads(sidecar.read_text())

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    window, patch_size, embed_dim, n_layers = _shape_from_state(state)

    # config.json is provenance, the tensors are ground truth — disagreement means
    # the .pt and the config are from different runs, which would silently
    # mislabel every plot.
    if cfg and cfg.get("n_time_steps") not in (None, window):
        raise typer.BadParameter(
            f"{ckpt}: config.json says window={cfg['n_time_steps']} but the weights "
            f"say {window}. Mismatched run directory."
        )

    return EncoderSpec(
        label=f"w{window}",
        path=ckpt,
        window=window,
        patch_size=patch_size,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=int(cfg.get("n_heads", default_heads)),
        norm=str(cfg.get("norm", "instance")),
    )


def load_encoder(spec: EncoderSpec, device: torch.device) -> JepaEncoder:
    enc = JepaEncoder(
        n_time_steps=spec.window,
        patch_size=spec.patch_size,
        embed_dim=spec.embed_dim,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        norm=spec.norm,
    ).to(device)
    enc.load_state_dict(
        torch.load(spec.path, map_location=device, weights_only=True), strict=True
    )
    enc.eval()
    return enc


# ============================================================================
#  DATA — glucose windows plus the labels we want to colour by
# ============================================================================

def _dataset_of(seq_id: str, patient_id: str) -> str:
    for prefix, name in DATASET_BY_PATIENT_PREFIX.items():
        if patient_id.startswith(prefix):
            return name
    return DATASET_BY_SEQ_PREFIX.get(seq_id.split("-")[0], "unknown")


def load_probe_frame(csv: Path, split: str, drop_interpolated: bool) -> pl.DataFrame:
    """One streaming pass: glucose plus BOTH id columns.

    `load_splits_streaming` keeps only the chosen unique_id, so it cannot give us
    sequence-keyed windows AND a patient label at the same time. Imputation is
    the shared policy (`impute_and_sort`), glucose-only.
    """
    lf = (
        pl.scan_csv(
            csv,
            infer_schema_length=10_000,
            schema_overrides={COL_SEQ: pl.Utf8, COL_USER: pl.Utf8},
        )
        .select([COL_SEQ, COL_USER, COL_TS, COL_SPLIT, COL_EVENT, COL_GLU])
        .rename({
            COL_SEQ: "unique_id",
            COL_USER: "patient_id",
            COL_TS: "ds",
            COL_SPLIT: "split",
            COL_EVENT: "event_type",
            COL_GLU: "glucose",
        })
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, TS_FORMAT, strict=False),
            pl.col("glucose").cast(pl.Float32, strict=False),
        ])
        .drop_nulls(subset=["unique_id", "patient_id", "ds", "split"])
    )
    if drop_interpolated:
        lf = lf.filter(pl.col("event_type") != "Interpolated")
    if split != "all":
        lf = lf.filter(pl.col("split") == split)

    df = lf.collect()
    if df.is_empty():
        raise typer.BadParameter(f"No rows for split={split!r} in {csv}")
    echo_plain(f"  {len(df):,} rows in split={split}")
    return impute_and_sort(df, ffill_bfill_columns=["glucose"])


@dataclass
class WindowSample:
    windows: np.ndarray      # (N, window) raw mg/dL
    dataset: np.ndarray      # (N,) "loop" | "aiready"
    patient: np.ndarray      # (N,) User ID
    mean_glucose: np.ndarray  # (N,)
    trend: np.ndarray        # (N,) the pretraining colouring, as a control


def sample_windows(
    df: pl.DataFrame,
    window: int,
    per_dataset: int,
    stride: int,
    rng: random.Random,
) -> WindowSample:
    """Balanced sample: `per_dataset` windows per dataset, spread evenly over that
    dataset's patients.

    Both levels of balancing matter. Loop and AI-READI contribute wildly
    different window counts, and an unbalanced sample makes PCA describe the
    larger cohort while the scatter's visual density says "separated" for what is
    really one dense blob and one sparse one. Within a dataset the same trap
    repeats per patient: Loop has few, long series, so a proportional draw hands
    one patient most of the cohort and `patient_silhouette` then measures that
    patient against everyone else rather than patient structure in general.
    """
    index: dict[str, dict[str, list[tuple[np.ndarray, int, str]]]] = {}
    n_short = 0

    for (seq_id,), grp in df.group_by(["unique_id"], maintain_order=True):
        g = grp["glucose"].to_numpy().astype(np.float32)
        n_windows = len(g) - window + 1
        if n_windows <= 0:
            n_short += 1
            continue
        patient = str(grp["patient_id"][0])
        ds = _dataset_of(str(seq_id), patient)
        starts = range(0, n_windows, stride)
        by_patient = index.setdefault(ds, {})
        by_patient.setdefault(patient, []).extend((g, s, patient) for s in starts)

    if n_short:
        echo_plain(f"  {n_short} series shorter than {window} steps contribute no windows")
    if not index:
        raise typer.BadParameter(f"No series reach {window} steps — nothing to probe.")

    chosen: list[tuple[np.ndarray, int, str]] = []
    labels: list[str] = []
    for ds in sorted(index):
        by_patient = index[ds]
        available = sum(len(v) for v in by_patient.values())
        # Round-robin over shuffled per-patient pools: everyone contributes their
        # first window before anyone contributes a second, so patients with long
        # records cannot swamp the draw.
        pools = [rng.sample(v, len(v)) for _, v in sorted(by_patient.items())]
        rng.shuffle(pools)
        picked: list[tuple[np.ndarray, int, str]] = []
        while len(picked) < per_dataset and pools:
            pools = [p for p in pools if p]
            for pool in pools:
                picked.append(pool.pop())
                if len(picked) >= per_dataset:
                    break
        chosen.extend(picked)
        labels.extend([ds] * len(picked))
        echo_plain(
            f"  {ds}: {len(picked):,} windows sampled from {available:,} available "
            f"across {len(by_patient)} patients"
        )

    # dataset_silhouette is only a *dataset* effect if each side has enough
    # patients to average over. Loop contributes a handful, so a thin side means
    # the number is partly measuring "these few people" — say so rather than let
    # it be read as cohort separation.
    thin = {ds: len(index[ds]) for ds in index if len(index[ds]) < 10}
    if thin:
        echo_plain(
            "  CAUTION: " + ", ".join(f"{ds} has only {n} patient(s)" for ds, n in thin.items())
            + " — dataset separation here is confounded with patient identity."
        )

    windows = np.stack([g[s : s + window] for g, s, _ in chosen]).astype(np.float32)
    return WindowSample(
        windows=windows,
        dataset=np.array(labels),
        patient=np.array([p for _, _, p in chosen]),
        mean_glucose=windows.mean(axis=1),
        trend=window_trend(windows),
    )


@torch.no_grad()
def encode(encoder: JepaEncoder, windows: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    """(N, window) -> (N, embed_dim), mean-pooled over patches like the training plots."""
    out = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(windows[i : i + batch]).to(device)
        out.append(encoder(x).float().mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================================
#  METRICS — so "the colours look separated" becomes a number
# ============================================================================

def effective_rank(x: np.ndarray) -> float:
    """Participation ratio of the covariance eigenvalues, (sum l)^2 / sum(l^2).

    Same definition `jepa_pretrain.collapse_metrics` logs — but computed here on
    the MEAN-POOLED embeddings, which is the space the silhouettes live in. The
    pretraining configs report it over per-patch tokens instead, and the two are
    not interchangeable: pooling low-passes the patch axis and drops the rank
    hard (36.0 over tokens vs 7.8 pooled, for the 864 encoder).
    """
    xc = x - x.mean(axis=0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(xc, rowvar=False)).clip(min=0)
    denom = (ev ** 2).sum()
    return float(ev.sum() ** 2 / denom) if denom > 0 else 0.0


def structure_metrics(pooled: np.ndarray, sample: WindowSample, n_pcs: int) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    metrics: dict[str, float] = {}
    pcs_full = PCA(n_components=min(pooled.shape)).fit_transform(pooled)

    # Silhouette is computed in the embedding, never in the 2-D scatter: PC1/PC2
    # are whatever carries the most variance, which need not be the axis a label
    # lives on. But the full 96-d space is mostly empty, and silhouette is
    # compressed toward 0 by distance concentration in the unused dimensions —
    # so report it BOTH at full width and truncated to the effective rank, which
    # is the width the cloud actually occupies.
    rank = effective_rank(pooled)
    k = max(2, min(int(round(rank)), pcs_full.shape[1]))
    metrics["eff_rank_pooled"] = rank
    metrics["k"] = float(k)
    trunc = pcs_full[:, :k]

    if len(set(sample.dataset)) > 1:
        metrics["dataset_silhouette_full"] = float(silhouette_score(pooled, sample.dataset))
        metrics["dataset_silhouette"] = float(silhouette_score(trunc, sample.dataset))

        # The best silhouette ANY 2-way split of this cloud reaches at the same
        # width. Truncation flatters every partition, not just the dataset one,
        # so the raw score above is only interpretable against this ceiling.
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(trunc)
        ceiling = float(silhouette_score(trunc, km))
        metrics["dataset_silhouette_ceiling"] = ceiling
        metrics["dataset_silhouette_ratio"] = (
            metrics["dataset_silhouette"] / ceiling if ceiling > 0 else float("nan")
        )

        # Separability with no distance geometry in it at all, so immune to the
        # dimensionality question the silhouettes are stuck with. Grouped by
        # patient: whole patients are held out, so it cannot memorise people.
        y = (sample.dataset == "loop").astype(int)
        n_groups = len(np.unique(sample.patient))
        if 0 < y.sum() < len(y) and n_groups >= 2:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            proba = cross_val_predict(
                clf, pooled, y, groups=sample.patient,
                cv=GroupKFold(n_splits=min(5, n_groups)), method="predict_proba",
            )[:, 1]
            metrics["dataset_probe_auc"] = float(roc_auc_score(y, proba))

    # Only patients with enough windows to be a cluster at all. Note this score
    # is pushed toward 0 by the sheer number of clusters — with ~350 patients the
    # nearest rival cluster is close by chance — so read it as weak evidence.
    ids, counts = np.unique(sample.patient, return_counts=True)
    keep = set(ids[counts >= 10])
    mask = np.array([p in keep for p in sample.patient])
    if mask.sum() > 10 and len(keep) > 1:
        metrics["patient_silhouette"] = float(silhouette_score(trunc[mask], sample.patient[mask]))
        metrics["patients_scored"] = float(len(keep))

    # How much of mean glucose survives the encoder's instance norm. Expected
    # near zero — that is the point of measuring it.
    pcs = pcs_full[:, :n_pcs]
    for name, target in (("glucose", sample.mean_glucose), ("trend", sample.trend)):
        model = LinearRegression().fit(pcs, target)
        metrics[f"{name}_r2"] = float(model.score(pcs, target))
    return metrics


# ============================================================================
#  PLOTTING
# ============================================================================

def _scatter_categorical(ax, xy, labels, colors, title, max_shown=None, rng=None):
    """Categorical scatter. With many categories, colour the biggest `max_shown`
    and grey the rest — 200 legend entries is not a plot."""
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts)]
    shown = list(order[:max_shown]) if max_shown else list(order)
    rest = np.array([lab not in shown for lab in labels])
    if rest.any():
        ax.scatter(xy[rest, 0], xy[rest, 1], s=6, alpha=0.25, color="#cccccc",
                   label=f"other ({len(order) - len(shown)})", linewidths=0)
    for i, lab in enumerate(shown):
        m = labels == lab
        c = colors[lab] if isinstance(colors, dict) else colors(i)
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.8, color=c,
                   label=f"{lab} ({m.sum()})", linewidths=0)
    ax.set_title(title)
    ax.legend(fontsize=6, markerscale=1.6, loc="best", framealpha=0.85)


def _scatter_continuous(ax, xy, values, label, title, fig, cmap="viridis", clip=(2, 98)):
    lo, hi = np.percentile(values, clip)   # outliers otherwise flatten the colour range
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=8, alpha=0.8,
                    vmin=lo, vmax=hi, linewidths=0)
    fig.colorbar(sc, ax=ax, label=label)
    ax.set_title(title)


def plot_encoder_pca(
    projected: np.ndarray,
    var: np.ndarray,
    sample: WindowSample,
    spec: EncoderSpec,
    metrics: dict[str, float],
    out_path: Path,
    max_patients: int,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = projected[:, :2]
    nan = float("nan")
    k = int(metrics.get("k", 0))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    days = spec.window * 5 / 60 / 24
    fig.suptitle(
        f"JEPA encoder {spec.label} — window {spec.window} steps ({days:.0f}d) · "
        f"{len(xy):,} windows · PC1 {var[0] * 100:.1f}% / PC2 {var[1] * 100:.1f}%\n"
        f"scatter = PC1/PC2; silhouettes measured in the top {k} PCs "
        f"(effective rank {metrics.get('eff_rank_pooled', float('nan')):.1f} of "
        f"{spec.embed_dim}), NOT in this 2-D view\n{spec.path}",
        fontsize=10,
    )

    _scatter_categorical(
        axes[0, 0], xy, sample.dataset, DATASET_COLORS,
        f"dataset — silhouette {metrics.get('dataset_silhouette', nan):.3f} "
        f"(ceiling {metrics.get('dataset_silhouette_ceiling', nan):.3f}, "
        f"{metrics.get('dataset_silhouette_ratio', nan):.0%} of it)\n"
        f"probe AUC {metrics.get('dataset_probe_auc', nan):.3f}  ·  "
        f"silhouette at full 96-d would read {metrics.get('dataset_silhouette_full', nan):.3f}",
    )
    _scatter_categorical(
        axes[0, 1], xy, sample.patient, plt.get_cmap("tab20"),
        f"patient_id, top {max_patients} by window count  "
        f"(silhouette {metrics.get('patient_silhouette', nan):.3f})",
        max_shown=max_patients,
    )
    _scatter_continuous(
        axes[1, 0], xy, sample.mean_glucose, "mean glucose (mg/dL)",
        f"mean glucose  (R² from 10 PCs = {metrics.get('glucose_r2', float('nan')):.3f})", fig,
    )
    _scatter_continuous(
        axes[1, 1], xy, sample.trend, "(last − first) / std",
        f"window trend — pretraining control  (R² = {metrics.get('trend_r2', float('nan')):.3f})",
        fig, cmap="coolwarm",
    )
    for ax in axes.ravel():
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_grid(results: list[dict], sample: WindowSample, out_path: Path, max_patients: int) -> Path:
    """One row per encoder, one column per colour variable — the window-length
    comparison, which is the whole reason for running three encoders."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.6 * n), squeeze=False)
    fig.suptitle(
        f"JEPA encoder latent PCA by pretraining window · {len(sample.dataset):,} "
        f"{'balanced ' if len(set(sample.dataset)) > 1 else ''}windows, identical sample across rows",
        fontsize=13,
    )
    for r, res in enumerate(results):
        spec, xy, var, m = res["spec"], res["projected"][:, :2], res["var"], res["metrics"]
        _scatter_categorical(axes[r][0], xy, sample.dataset, DATASET_COLORS,
                             f"{spec.label} · dataset (sil {m.get('dataset_silhouette', float('nan')):.3f}"
                             f" @{int(m.get('k', 0))} PCs, AUC {m.get('dataset_probe_auc', float('nan')):.3f})")
        _scatter_categorical(axes[r][1], xy, sample.patient, plt.get_cmap("tab20"),
                             f"{spec.label} · patient (sil {m.get('patient_silhouette', float('nan')):.3f})",
                             max_shown=max_patients)
        _scatter_continuous(axes[r][2], xy, sample.mean_glucose, "mean glucose",
                            f"{spec.label} · glucose (R² {m.get('glucose_r2', float('nan')):.3f})", fig)
        for c in range(3):
            axes[r][c].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
            axes[r][c].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ============================================================================
#  CLI
# ============================================================================

@app.command()
def main(
    encoder: list[str] = typer.Option(
        list(DEFAULT_ENCODERS), "--encoder",
        help="Run directory or encoder .pt. Repeatable; defaults to the 288/864/2016 runs.",
    ),
    csv: Path = typer.Option(
        Path("data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv"), help="ML-ready CSV."
    ),
    split: str = typer.Option(
        "val", help="train | val | test | all. Default val = series pretraining never saw.",
    ),
    prefer: str = typer.Option("best", help="best | last — which encoder*.pt to take from a run dir."),
    windows_per_dataset: int = typer.Option(2000, help="Windows sampled per dataset (balanced)."),
    window_stride: int = typer.Option(12, help="Stride of the candidate window grid, in steps."),
    max_patients: int = typer.Option(12, help="Patients given their own colour; the rest go grey."),
    n_pcs: int = typer.Option(10, help="PCs used for the glucose/trend R² regressions."),
    drop_interpolated: bool = typer.Option(False, help="Drop Interpolated rows."),
    batch_size: int = typer.Option(256, help="Encoding batch size."),
    device_name: str = typer.Option("cuda", "--device", help="cpu | mps | cuda."),
    seed: int = typer.Option(42, help="Random seed for the window sample."),
    out_dir: Path = typer.Option(Path("data/output/encoder_pca"), help="Output directory."),
) -> None:
    """PCA of pretrained JEPA encoder latents, coloured by dataset / patient / glucose."""
    if device_name == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    specs = [resolve_encoder(e, prefer, default_heads=6) for e in encoder]
    typer.echo(f"Device: {device} | {len(specs)} encoder(s)")
    for s in specs:
        typer.echo(
            f"  {s.label}: window={s.window} patch={s.patch_size} dim={s.embed_dim} "
            f"layers={s.n_layers} heads={s.n_heads} norm={s.norm}  <- {s.path}"
        )

    typer.echo(f"\nLoading {csv} (split={split})...")
    df = load_probe_frame(csv, split, drop_interpolated)

    # The sample is per-window-length: a 2016-step window cannot be drawn from
    # the same series as a 288-step one. Each encoder gets its own sample, drawn
    # from the same seeded RNG and balanced the same way.
    from sklearn.decomposition import PCA

    results = []
    for spec in specs:
        typer.echo(f"\n=== {spec.label} ===")
        sample = sample_windows(df, spec.window, windows_per_dataset, window_stride,
                                random.Random(seed))
        encoder_module = load_encoder(spec, device)
        pooled = encode(encoder_module, sample.windows, device, batch_size)
        echo_plain(f"  pooled latents {pooled.shape}")

        pca = PCA(n_components=min(spec.embed_dim, len(pooled), 20)).fit(pooled)
        metrics = structure_metrics(pooled, sample, n_pcs)
        for k, v in metrics.items():
            echo_plain(f"  {k} = {v:.4f}")

        results.append({
            "spec": spec,
            "sample": sample,
            "projected": pca.transform(pooled),
            "var": pca.explained_variance_ratio_,
            "metrics": metrics,
        })
        path = plot_encoder_pca(
            results[-1]["projected"], results[-1]["var"], sample, spec, metrics,
            out_dir / f"pca_{spec.label}.png", max_patients,
        )
        typer.echo(f"  wrote {path}")

    rows = [
        {
            "encoder": r["spec"].label,
            "window": r["spec"].window,
            "checkpoint": str(r["spec"].path),
            "n_windows": len(r["sample"].dataset),
            "n_patients": int(len(np.unique(r["sample"].patient))),
            "pc1_var": round(float(r["var"][0]), 4),
            "pc2_var": round(float(r["var"][1]), 4),
            **{k: round(v, 4) for k, v in r["metrics"].items()},
        }
        for r in results
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "probe_metrics.csv"
    pl.DataFrame(rows).write_csv(metrics_csv)
    (out_dir / "probe_config.json").write_text(json.dumps({
        "csv": str(csv), "split": split, "prefer": prefer,
        "windows_per_dataset": windows_per_dataset, "window_stride": window_stride,
        "n_pcs": n_pcs, "drop_interpolated": drop_interpolated, "seed": seed,
        "encoders": [str(s.path) for s in specs],
    }, indent=2))
    typer.echo(f"\nwrote {metrics_csv}")

    # The grid shares one sample across rows, which only holds if every encoder
    # drew the same one — true when all windows are equal length, not otherwise.
    if len({s.window for s in specs}) == 1 and len(results) > 1:
        typer.echo(f"wrote {plot_grid(results, results[0]['sample'], out_dir / 'pca_grid.png', max_patients)}")
    elif len(results) > 1:
        typer.echo(
            "Note: no combined grid — the encoders have different window lengths, so each "
            "row would be a different window sample and the columns would not be comparable. "
            "Compare the per-encoder PNGs and probe_metrics.csv."
        )


if __name__ == "__main__":
    app()
