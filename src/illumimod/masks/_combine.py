from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt

def combine(
    masks: Sequence[npt.NDArray],
    weights: Optional[Sequence[float]] = None,
    dtype = np.float32,
) -> npt.NDArray:
    """
    Additively combine multiple masks of the same spatial size.

    Parameters
    ----------
    masks : sequence of ndarray
        List/tuple of masks with shape ``(H, W)``. ``None`` entries are ignored.
    weights : sequence of float, optional
        Per-mask multipliers. Must be the same length as ``masks`` if provided.
    dtype : numpy dtype, optional
        Accumulation and output dtype. Defaults to ``np.float32``.

    Returns
    -------
    ndarray
        The additive sum with shape ``(H, W)`` and dtype ``dtype``.

    Raises
    ------
    ValueError
        If no masks are provided, shapes differ, or ``weights`` length mismatches.

    Notes
    -----
    This function performs **no normalization or clipping**; it preserves relative
    intensities exactly.
    """

    # filter out Nones
    ms = [m for m in masks if m is not None]
    if not ms:
        raise ValueError("combine_masks: no masks provided")

    h, w = ms[0].shape[:2]
    for m in ms:
        if m.shape[:2] != (h, w):
            raise ValueError("combine_masks: all masks must have the same HxW")

    out = np.zeros((h, w), dtype=dtype)

    if weights is None:
        for m in ms:
            out += m.astype(dtype, copy=False)
    else:
        if len(weights) != len(masks):
            raise ValueError("combine_masks: weights length must match masks length")
        for m, w in zip(masks, weights):
            if m is None:
                continue
            out += float(w) * m.astype(dtype, copy=False)

    return out