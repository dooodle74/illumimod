# masks.py (3.8/3.9-friendly)
from typing import Tuple, Optional, Sequence
import numpy as np
import numpy.typing as npt

def generate_light(
    width: int,
    height: int,
    center: Tuple[float, float],
    peak: float,
    core_radius: float,
    half_life_radius: float,
    power: float = 2.0,          # tail steepness: 1=softer/longer, 2≈gaussian-ish, >2 sharper
    edge_softness: float = 0.0,  # soften the core edge (px; 0 = hard edge)
    angle_deg: float = 90.0,     # 90 = perpendicular; smaller = grazing
    direction_deg: float = 0.0,  # 0=from up (north), 90=from right, clockwise
    lambertian: bool = False,    # if True, peak *= sin(angle) (optional physical dimming)
    dtype = np.float32,
) -> npt.NDArray:
    """
    Flat-top directional light with exact half-life from the core edge.

    Effective radius uses an ellipse stretched along the approach direction
    by 1/sin(angle). Intensity profile:
        I = peak                                         for r_eff <= core_radius
            peak * 2^{-((r_eff - core_radius)/half_life_radius)^power}  otherwise
    ⇒ I(core_radius + half_life_radius) = peak/2  (by construction).
    """
    w, h = int(width), int(height)
    cx, cy = float(center[0]), float(center[1])

    # coordinate grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    # approach direction unit vector (image y points down):
    # direction_deg measured FROM NORTH (up), clockwise.
    phi = np.deg2rad(direction_deg)
    t_par = np.array([np.sin(phi), -np.cos(phi)], dtype=np.float32)  # along approach
    t_perp = np.array([-t_par[1], t_par[0]], dtype=np.float32)       # rotate +90°

    # project into (parallel, perpendicular) axes
    proj_par  = dx * t_par[0]  + dy * t_par[1]
    proj_perp = dx * t_perp[0] + dy * t_perp[1]

    # ellipse stretch along approach due to grazing incidence
    ang = np.deg2rad(max(min(angle_deg, 90.0), 1e-3))  # clamp away from 0
    stretch = 1.0 / max(np.sin(ang), 1e-6)             # 1 at 90°, grows as angle→0

    R0 = float(core_radius)
    H  = max(float(half_life_radius), 1e-6)
    p  = max(float(power), 1e-6)

    # elliptical effective radius
    r_eff = np.sqrt((proj_par / stretch) ** 2 + (proj_perp) ** 2)

    # base peak (optional Lambertian dimming with angle)
    I0 = float(peak) * (float(np.sin(ang)) if lambertian else 1.0)

    # flat core + half-life tail
    dr = np.maximum(0.0, r_eff - R0)
    fall = 2.0 ** ( - (dr / H) ** p )
    I = I0 * fall
    I[r_eff <= R0] = I0

    # optional soft core edge (smoothstep across [R0 - s, R0 + s])
    s_edge = max(float(edge_softness), 0.0)
    if s_edge > 0.0:
        a0, a1 = R0 - s_edge, R0 + s_edge
        denom = max(a1 - a0, 1e-6)
        t = np.clip((r_eff - a0) / denom, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        I = (1.0 - t) * I0 + t * I

    return I.astype(dtype, copy=False)

def combine(
    masks: Sequence[npt.NDArray],
    weights: Optional[Sequence[float]] = None,
    dtype = np.float32,
) -> npt.NDArray:
    """
    Purely additive combination of a list of (H,W) masks.

    Parameters
    ----------
    masks : sequence of ndarray
        Each mask must be shape-compatible (same H,W). None entries are ignored.
    weights : sequence of float or None
        Optional per-mask weights (same length as masks). Default = 1.0 for all.
    dtype : numpy dtype
        Output dtype (and accumulation dtype). Default float32.

    Returns
    -------
    ndarray (H,W)
        Sum_i weights[i] * masks[i], no clipping or normalization.
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
    Linearly map the interval [P_low(mask), P_high(mask)] -> [low, high].
    - If low/high are None, they default to those same percentiles,
      making the transform an identity (no change).
    - Set clip=True to clamp everything outside [low, high] after mapping.
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
    Auto-fit a mask to optional [low, high] bounds using linear scaling
    with robust anchors. No-op unless a bound is out of range.

    Behavior:
      - If both bounds are violated → scale(low, high)
      - If only high is violated   → scale(original_low, high)
      - If only low is violated    → scale(low, original_high)
      - If neither is violated or both bounds are None → return unchanged

    “original_low/high” are the mask’s P_low / P_high percentiles (defaults 1/99).
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

def apply(
    img: npt.NDArray,           # (H,W) or (H,W,3)
    mask: npt.NDArray,          # (H,W), same spatial size as img
    *,
    exposure: float = 1.0,
    clip: bool = True,
    out_dtype = np.uint8,
) -> npt.NDArray:
    """
    Add the mask to the image as-is (no auto-scaling), then clip.

    Assumptions:
      - mask is already in the correct units (e.g., 0..255 if img is uint8 sRGB).
      - mask is non-negative (lights add). Negative values will darken.

    Returns image after mask.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("apply_additive: mask HxW must match image HxW")

    img_f  = img.astype(np.float32, copy=False)
    add    = mask.astype(np.float32, copy=False)

    # Broadcast to RGB if needed
    if img_f.ndim == 3 and img_f.shape[2] != 1 and add.ndim == 2:
        add = add[..., None]

    out = exposure * img_f + add

    if clip:
        headroom = 255.0 if np.issubdtype(img.dtype, np.integer) else 1.0
        out = np.clip(out, 0.0, headroom)

    return out.astype(out_dtype, copy=False)