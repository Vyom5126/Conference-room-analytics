"""
ood.py
"Does this look like the photos the model was trained on?" check.

similarity(e) = mean cosine similarity between an image embedding and its k nearest
training embeddings. The threshold is a low percentile of the same score computed
leave-one-out on the training photos, so roughly that fraction of genuine training
photos would be flagged. Photos below it get a warning instead of a confident score.
"""
import numpy as np

K = 5


def knn_similarity(e, ref, k=K, exclude_self=False):
    """Mean cosine similarity of each row of e (L2-normalised) to its k nearest rows of ref."""
    e = np.atleast_2d(e)
    sims = e @ ref.T
    if exclude_self:   # e is ref: drop each row's similarity to itself
        np.fill_diagonal(sims, -np.inf)
    top = -np.sort(-sims, axis=1)[:, :k]
    return top.mean(axis=1)


def fit_reference(ref, percentile=1.0, k=K):
    """Leave-one-out similarities of the training photos and the flagging threshold."""
    loo = knn_similarity(ref, ref, k, exclude_self=True)
    return {"threshold": float(np.percentile(loo, percentile)), "percentile": percentile, "k": k,
            "loo_min": float(loo.min()), "loo_median": float(np.median(loo))}
