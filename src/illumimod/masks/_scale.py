from typing import Optional
import numpy as np
import numpy.typing as npt

def scale(
    mask: npt.NDArray,
    low: Optional[float] = None,            # default: use p1 of the mask
    high: Optional[float] = None,           # default: use p99 of the mask
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    *,
    clip: bool = False,                     # default False so identity holds when low/high are None
    dtype = np.float32,
    eps: float = 1e-6,
    return_stats: bool = False,
) -> npt.NDArray:
    """
    Linearly map two percentiles of a mask to target values.

    Maps ``[P_low(mask), P_high(mask)] → [low, high]`` and linearly rescales in between.
    If ``low``/``high`` are ``None``, they default to those same percentiles, making the
    transform an **identity** by default. Optionally clamps values after mapping.

    Parameters
    ----------
    mask : ndarray
        Input mask ``(H, W)``.
    low : float or None, optional
        Target value for ``P_low(mask)``. If ``None``, uses the input percentile itself.
    high : float or None, optional
        Target value for ``P_high(mask)``. If ``None``, uses the input percentile itself.
    low_percentile : float, optional
        Lower anchor percentile (e.g., ``1.0``).
    high_percentile : float, optional
        Upper anchor percentile (e.g., ``99.0``).
    clip : bool, optional
        If ``True``, clamp outputs into ``[min(low,high), max(low,high)]`` after mapping.
    dtype : numpy dtype, optional
        Output dtype. Defaults to ``np.float32``.
    eps : float, optional
        Small constant to guard near-zero spans.
    return_stats : bool, optional
        Reserved for future use; ignored in this version.

    Returns
    -------
    ndarray
        Rescaled mask with dtype ``dtype``.

    Notes
    -----
    - Supports inverted targets (``low > high``); clamping honors the numeric order.
    - If the selected percentile span is ~0, outputs the midpoint ``(low+high)/2``.
    """
    m = mask.astype(np.float32, copy=False)
    p_lo = float(np.percentile(m, float(low_percentile)))
    p_hi = float(np.percentile(m, float(high_percentile)))

    lo_t = p_lo if low  is None else float(low)
    hi_t = p_hi if high is None else float(high)

    span = p_hi - p_lo
    if abs(span) < eps:
        # Nearly constant input in the selected band.
        out = np.full_like(m, (lo_t + hi_t) * 0.5, dtype=np.float32)
    else:
        norm = (m - p_lo) / span
        out  = lo_t + norm * (hi_t - lo_t)
        if clip:
            lo_c, hi_c = (lo_t, hi_t) if lo_t <= hi_t else (hi_t, lo_t)
            out = np.clip(out, lo_c, hi_c)

    out = out.astype(dtype, copy=False)
    return out

def autocap(
    mask: npt.NDArray,
    low: Optional[float] = None,            # lower bound to enforce (e.g., 0.0); None = ignore
    high: Optional[float] = None,           # upper bound to enforce (e.g., 255.0); None = ignore
    *,
    low_percentile: float = 1.0,            # anchors for “original low/high”
    high_percentile: float = 99.0,
    clip: bool = True,                      # clamp to [low, high] when rescaling
    dtype = np.float32,
    eps: float = 1e-6,
) -> npt.NDArray:
    """
    Auto-fit a mask to optional bounds using robust anchors, or return it unchanged.

    Decides whether to rescale based on the mask’s robust range
    ``[P_low(mask), P_high(mask)]``:

    - If **both** bounds are violated → scale to ``[low, high]``.
    - If only **high** is violated   → scale ``[P_low, P_high] → [P_low, high]``.
    - If only **low** is violated    → scale ``[P_low, P_high] → [low, P_high]``.
    - If neither bound is violated (or both bounds are ``None``) → no-op.

    Parameters
    ----------
    mask : ndarray
        Input mask ``(H, W)``.
    low : float or None, optional
        Lower bound to enforce (e.g., ``0.0``). ``None`` = ignore lower bound.
    high : float or None, optional
        Upper bound to enforce (e.g., ``255.0``). ``None`` = ignore upper bound.
    low_percentile : float, optional
        Robust low anchor percentile used to detect/anchor the lower end (default ``1.0``).
    high_percentile : float, optional
        Robust high anchor percentile used to detect/anchor the upper end (default ``99.0``).
    clip : bool, optional
        If rescaling occurs, clamp outputs into ``[low, high]`` (or the appropriate half-bound).
    dtype : numpy dtype, optional
        Output dtype. Defaults to ``np.float32``.
    eps : float, optional
        Tolerance for bound checks.

    Returns
    -------
    ndarray
        Either the original mask (no-op) or a linearly rescaled mask with dtype ``dtype``.
    """

    m = mask.astype(np.float32, copy=False)
    if m.size == 0 or (low is None and high is None):
        return m.astype(dtype, copy=False)

    # Robust anchors from the current mask
    p_lo = float(np.percentile(m, float(low_percentile)))
    p_hi = float(np.percentile(m, float(high_percentile)))

    # Which bounds are actually out of range (robustly)?
    low_out  = (low  is not None) and (p_lo < float(low)  - eps)
    high_out = (high is not None) and (p_hi > float(high) + eps)

    if not (low_out or high_out):
        # Already within requested bounds → no change
        return m.astype(dtype, copy=False)

    # Decide target mapping
    # Ignore type check - to get to these ifs low and high must not be None
    if low_out and high_out:
        # Fit both ends
        return scale(
            m,
            low=float(low), high=float(high),
            low_percentile=low_percentile, high_percentile=high_percentile,
            clip=clip, dtype=dtype,
        )
    elif high_out:
        # Preserve original low, fit high
        return scale(
            m,
            low=p_lo, high=float(high),
            low_percentile=low_percentile, high_percentile=high_percentile,
            clip=clip, dtype=dtype,
        )
    else:  # low_out only
        return scale(
            m,
            low=float(low), high=p_hi,
            low_percentile=low_percentile, high_percentile=high_percentile,
            clip=clip, dtype=dtype,
        )
