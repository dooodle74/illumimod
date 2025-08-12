# masks.py (3.8/3.9-friendly)
from typing import Tuple, Optional, Sequence
import numpy as np
import numpy.typing as npt

def light_to_mask(
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

def scale_mask(
    mask: npt.NDArray,
    low: float,
    high: float,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    *,
    clip: bool = True,
    dtype = np.float32,
    eps: float = 1e-6,
) -> npt.NDArray:
    """
    Linearly scale a mask so that:
      P_low(mask)  -> low
      P_high(mask) -> high
    Everything below/above is optionally clipped.

    Example: scale_mask(L, 0.0, 1.0)  # robust normalize to [0,1]
    """
    m = mask.astype(np.float32, copy=False)

    p_lo = float(np.percentile(m, float(low_percentile)))
    p_hi = float(np.percentile(m, float(high_percentile)))

    # Degenerate case: almost no spread between chosen percentiles
    if (p_hi - p_lo) < eps:
        out = np.full_like(m, (low + high) * 0.5, dtype=np.float32)
    else:
        norm = (m - p_lo) / (p_hi - p_lo)
        out = low + norm * (high - low)
        if clip:
            if low <= high:
                out = np.clip(out, low, high)
            else:
                # inverted ranges
                out = np.clip(out, high, low)

    out = out.astype(dtype, copy=False)

    return out

def combine_masks(
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