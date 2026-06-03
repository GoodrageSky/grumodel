"""
embed_finbert_bertopic.py
=========================
Stage 2 of the daily-frequency text-factor pipeline (your idea #2).

text  --FinBERT-->  768-d embedding  --supervised UMAP / BERTopic-->  low-d feature panel

Two design points that matter for a quant factor:

* LEAKAGE: supervised dimensionality reduction uses the LABEL. The reducer is
  therefore fit ONLY on the training period and merely *transforms* valid/test.
  Discrete bins of the forward-return rank are used as the supervision target.

* OUTPUT CONTRACT: the reduced dimensions are written out as columns named
  `emb_0 ... emb_{k-1}` indexed by [date, stkcd], i.e. exactly the format the
  Stage-3 GRU+LGBM module already consumes. So Stage 2 is a drop-in replacement
  for whatever produced the original `dt_lgbm.parquet`.

For Chinese A-share text the default FinBERT is 熵简科技 `valuesimplex/FinBERT`.
If you care about *embedding* quality specifically (FinBERT is a masked-LM, not
a sentence encoder), a retrieval-tuned model such as `BAAI/bge-large-zh-v1.5`
or FinBERT2 / Fin-Retriever usually gives cleaner vectors — pass its name to
`FinBertEmbedder(model_name=...)`.

All heavy imports (torch / transformers / umap / bertopic) are lazy so this file
imports even when those packages are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# FinBERT embedder
# --------------------------------------------------------------------------- #
@dataclass
class EmbedConfig:
    model_name: str = "valuesimplex/FinBERT"
    max_length: int = 512
    batch_size: int = 64
    pooling: str = "mean"          # "mean" (recommended for MLM-style BERT) or "cls"
    device: str = "auto"
    normalize: bool = True         # L2-normalise output vectors


class FinBertEmbedder:
    """Mean-pooled (attention-masked) FinBERT sentence embeddings."""

    def __init__(self, cfg: EmbedConfig = EmbedConfig()):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if cfg.device == "auto" else cfg.device
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(self.device).eval()

    def _pool(self, last_hidden, attn_mask):
        import torch
        if self.cfg.pooling == "cls":
            return last_hidden[:, 0]
        mask = attn_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        vecs = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = [t if isinstance(t, str) and t.strip() else "无" for t in texts[i:i + self.cfg.batch_size]]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            pooled = self._pool(out, enc["attention_mask"])
            if self.cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=1)
            vecs.append(pooled.cpu().numpy())
        return np.concatenate(vecs).astype(np.float32)


def embed_panel(df: pd.DataFrame, embedder: FinBertEmbedder, text_col: str = "summary") -> np.ndarray:
    """Embed one text column of a [date, stkcd] panel; rows stay aligned to df."""
    return embedder.encode(df[text_col].fillna("").tolist())


# --------------------------------------------------------------------------- #
# Supervised dimensionality reduction
# --------------------------------------------------------------------------- #
@dataclass
class ReduceConfig:
    n_components: int = 10         # output feature dimensions (-> emb_0..emb_{k-1})
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"
    n_bins: int = 10               # quantile bins of the label for supervision
    random_state: int = 42
    use_bertopic: bool = True      # also fit a BERTopic model for topic features
    nr_topics: Optional[int] = None  # None = auto; or an int to merge to k topics


def _label_to_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretise a continuous label (e.g. forward-return rank in [0,1]) into
    integer classes for supervised UMAP. NaNs -> -1 (UMAP treats -1 as 'unknown',
    i.e. unsupervised for that point)."""
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), -1, dtype=int)
    ok = ~np.isnan(y)
    if ok.sum() == 0:
        return out
    ranks = pd.Series(y[ok]).rank(method="first")
    out[ok] = pd.qcut(ranks, min(n_bins, max(ok.sum() // 2, 1)),
                      labels=False, duplicates="drop").to_numpy()
    return out


class SupervisedReducer:
    """Supervised UMAP fit on the TRAIN split only, plus an optional BERTopic
    model that reuses the same supervised UMAP. Produces:
      * reduced embeddings  (N, n_components)  -> the `emb_*` features
      * topic ids / probabilities             -> optional extra features
    """

    def __init__(self, cfg: ReduceConfig = ReduceConfig()):
        self.cfg = cfg
        self.umap_ = None
        self.topic_model_ = None

    def fit(self, embeddings_train: np.ndarray, y_train: np.ndarray,
            docs_train: Optional[list[str]] = None):
        import umap
        y_bins = _label_to_bins(y_train, self.cfg.n_bins)
        self.umap_ = umap.UMAP(
            n_components=self.cfg.n_components, n_neighbors=self.cfg.n_neighbors,
            min_dist=self.cfg.min_dist, metric=self.cfg.metric,
            random_state=self.cfg.random_state,
        ).fit(embeddings_train, y=y_bins)        # <-- supervised: y steers the manifold

        if self.cfg.use_bertopic and docs_train is not None:
            from bertopic import BERTopic
            # Reuse the already-fitted supervised UMAP so topics inherit the
            # label-aware geometry; pass precomputed embeddings to skip re-encoding.
            self.topic_model_ = BERTopic(
                umap_model=self.umap_, nr_topics=self.cfg.nr_topics,
                calculate_probabilities=True, verbose=False,
            )
            self.topic_model_.fit(docs_train, embeddings=embeddings_train, y=y_bins)
        return self

    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduced features for any split (train/valid/test) — transform only."""
        assert self.umap_ is not None, "call fit() first"
        return self.umap_.transform(embeddings).astype(np.float32)

    def transform_topics(self, docs: list[str], embeddings: np.ndarray):
        """Topic ids + probability matrix for any split, if BERTopic was fit."""
        assert self.topic_model_ is not None, "use_bertopic was False or fit() not called"
        topics, probs = self.topic_model_.transform(docs, embeddings=embeddings)
        return np.asarray(topics), np.asarray(probs)


# --------------------------------------------------------------------------- #
# Assemble the feature panel that Stage 3 consumes
# --------------------------------------------------------------------------- #
def build_feature_panel(df_daily: pd.DataFrame,
                        reduced: np.ndarray,
                        label_series: Optional[pd.Series] = None,
                        return_series: Optional[pd.Series] = None,
                        prefix: str = "emb_") -> pd.DataFrame:
    """Combine reduced features with the [date, stkcd] index into the panel
    format Stage 3 expects (columns emb_0.. plus label/return)."""
    cols = [f"{prefix}{i}" for i in range(reduced.shape[1])]
    out = pd.DataFrame(reduced, columns=cols, index=df_daily.index)
    if label_series is not None:
        out = out.join(label_series.rename(label_series.name or "vwap_rk"))
    if return_series is not None:
        out = out.join(return_series.rename(return_series.name or "vwap_ret20"))
    return out.sort_index()


def split_by_segments(df: pd.DataFrame, segments: dict, date_level: int = 0):
    """Return boolean masks (aligned to df rows) for named date segments."""
    dates = df.index.get_level_values(date_level)
    masks = {}
    for name, sl in segments.items():
        masks[name] = (dates >= pd.Timestamp(sl.start)) & (dates <= pd.Timestamp(sl.stop))
    return masks
