"""Visualize the post-reduction text-embedding panel.

Reads artifacts/panel.parquet (emb_0..emb_{k-1} + vwap_rk + vwap_ret20)
and shows a 2x2 figure:

  TL  2D PCA of emb_* colored by vwap_rk quintile (does the reducer separate?)
  TR  |Pearson(emb_i, vwap_rk)| per dim, split train vs test (which dims carry signal?)
  BL  per-dim violin (scale / outliers)
  BR  cross-sectional IC time series of the strongest dim (temporal stability)

Also saves the figure to artifacts/viz/panel.png.
"""
from __future__ import annotations
import sys
from pathlib import Path

# project layout: pipelines/<this file> + src/<modules>. Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import PATHS, SEGMENTS


def _segment_of(date_idx: pd.Index) -> pd.Series:
    """Tag each date as train / valid / test / out (per SEGMENTS)."""
    s = pd.Series("out", index=date_idx, dtype=object)
    for name, sl in SEGMENTS.items():
        m = (date_idx >= pd.Timestamp(sl.start)) & (date_idx <= pd.Timestamp(sl.stop))
        s[m] = name
    return s


def _per_date_ic(x: pd.Series, y: pd.Series) -> pd.Series:
    """Cross-sectional Pearson correlation per date."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return df.groupby(level=0).apply(
        lambda g: g["x"].corr(g["y"]) if len(g) >= 5 else np.nan
    )


def main() -> None:
    panel = pd.read_parquet(PATHS["panel"])
    emb_cols = [c for c in panel.columns if c.startswith("emb_")]
    print(f"loaded panel: {panel.shape} | emb dims: {len(emb_cols)}")

    X = panel[emb_cols].to_numpy()
    y = panel["vwap_rk"].to_numpy()
    dates = panel.index.get_level_values(0)
    seg = _segment_of(dates)

    # === TL: 2D PCA colored by vwap_rk quintile =============================
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    quintile = pd.qcut(panel["vwap_rk"], 5, labels=False, duplicates="drop")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Text-embedding panel · {len(panel):,} rows · "
        f"{len(emb_cols)} dims (post supervised UMAP)",
        fontsize=14,
    )

    ax = axes[0, 0]
    sc = ax.scatter(XY[:, 0], XY[:, 1], c=quintile, cmap="RdYlGn",
                    s=6, alpha=0.55, linewidths=0)
    ax.set_title("PCA(10D → 2D) colored by future-return quintile\n"
                 "(green = top quintile, red = bottom)")
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%})")
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("vwap_rk quintile (0=bottom, 4=top)")

    # === TR: |corr(emb_i, vwap_rk)| per dim, train vs test ==================
    ax = axes[0, 1]
    df = panel.copy()
    df["seg"] = seg.values
    corrs = {}
    for sname in ("train", "test"):
        sub = df[df["seg"] == sname]
        if sub.empty:
            corrs[sname] = np.zeros(len(emb_cols))
            continue
        corrs[sname] = np.array(
            [abs(sub[c].corr(sub["vwap_rk"])) for c in emb_cols]
        )
    width = 0.4
    xs = np.arange(len(emb_cols))
    ax.bar(xs - width / 2, corrs["train"], width, label="train", color="#1f77b4")
    ax.bar(xs + width / 2, corrs["test"], width, label="test", color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels(emb_cols, rotation=45, ha="right")
    ax.set_ylabel("|Pearson(emb_i, vwap_rk)|")
    ax.set_title("Per-dim correlation with future-return rank")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # === BL: per-dim distribution (violin) ==================================
    ax = axes[1, 0]
    parts = ax.violinplot([panel[c].dropna().values for c in emb_cols],
                          showmeans=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.65)
    ax.set_xticks(range(1, len(emb_cols) + 1))
    ax.set_xticklabels(emb_cols, rotation=45, ha="right")
    ax.set_ylabel("value")
    ax.set_title("Per-dim distribution (scale / outliers)")
    ax.grid(True, axis="y", alpha=0.3)

    # === BR: cross-sectional IC time series of the strongest dim ============
    ax = axes[1, 1]
    strongest = emb_cols[int(np.argmax(corrs["train"]))]
    ic_ts = _per_date_ic(panel[strongest], panel["vwap_rk"])
    ic_ts.index = pd.to_datetime(ic_ts.index)
    # color points by segment
    seg_per_date = _segment_of(ic_ts.index)
    palette = {"train": "#1f77b4", "valid": "#2ca02c", "test": "#d62728", "out": "#888"}
    for sname, color in palette.items():
        mask = seg_per_date == sname
        if mask.any():
            ax.scatter(ic_ts.index[mask], ic_ts.values[mask],
                       s=22, color=color, label=sname, alpha=0.75)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
    ax.set_title(f"Cross-sectional IC over time  ({strongest} vs vwap_rk)")
    ax.set_ylabel("daily IC")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = PATHS["raw_docs"].parent / "viz" / "panel.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
