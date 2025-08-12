'''
TODO:
1. install lowest python version (pref. 3.8) venv, specify in reqs
2. Change behavior of angled light (now, x degrees is the same as (x+180)%360 degrees. Shouldn't be symmetric)
    - In the future, add back this accidental feature: ellipse light
3. Mask normalization. Focus on additive only? Not subtract. A few ideas
    - if purely additive, normalize to strictly [0, 255] for standard viewing
    - if purely subtractive, can't even view innit
        - maybe set 0 to full white on the png, then normalize to [-255, 0] - essentially "lifting up"
    - perhaps normalizing masks only make sense for viewing. Shouldn't even be needed. The information on the brightest/darkest pixels should be preserved - it is up to the light souce adder to make it make sense
4. Mask resizing. I understand that resizing down is easier
5. apply-time normalization. give users options
    - ONlY ADDITIVE FOR NOW, BUILD IN FUTUREPROOFING
    - dynamic-scaling: scale the mask range to the given headroom
        - specify upper bound for the mask to dynamically adjust to. if want maximum contrast, =255. 
        - potential problem scenario: there is already a very bright point (sun, close to 255) in an otherwise dark image, we want to add another bright point somewhere entirely else. using the above dynamic adjust, it simply wouldn't be added, even though the darker to-be-added area won't be close to 255 after adding. This is a non-issue for speckle images but will be an issue later for these actual images. 
    - zeroing/shifting: shift the mask to min = 0
        - FUTURE: for both signs, max=0, or median =0, any percent to 0
    - FUTURE: 
        - dynamic-signs-scales: keep mask 0 at 0, scale positive and negative seperately given headroom and limits
6. Research version control and naming convent
'''



# --- optional helpers for lookups (normalize keys) ---

def normalize_stat_key(key: Any) -> str:
    """
    Map various aliases to the fixed keys above.
      - numbers -> percentiles (25 -> 'p25', 99 -> 'p99')
      - '25%', 'p25', 'P25', 'p01' -> 'p25' / 'p1'
      - 'min'/'maximum' -> 'p0'/'p100'
      - 'median'/'med'  -> 'p50'
      - 'avg'/'average' -> 'mean'
      - 'stdev'/'sigma' -> 'std'
    """
    if isinstance(key, (int, float)):
        q = int(round(float(key)))
        return f"p{q}"

    s = str(key).strip().lower()
    if s.endswith("%"):
        s = s[:-1].strip()
    if s.startswith("p"):
        s = s[1:]
    if s in ("min", "minimum"):  return "p0"
    if s in ("max", "maximum"):  return "p100"
    if s in ("median", "med"):   return "p50"
    if s in ("avg", "average"):  return "mean"
    if s in ("stdev", "stddev", "σ", "sigma"): return "std"

    # percentiles like "01" or "1" -> p1
    try:
        q = int(round(float(s)))
        return f"p{q}"
    except ValueError:
        pass

    # fall through: allow exact fixed keys if already correct
    if s in {"mean","std","p0","p1","p25","p50","p75","p99","p100"}:
        return s
    raise KeyError("Unknown stat key: %r" % key)

def get_stat(stats: Dict[str, float], key: Any, default: Optional[float] = None) -> Optional[float]:
    """Convenience accessor that normalizes the key first."""
    try:
        k = normalize_stat_key(key)
    except KeyError:
        return default
    return stats.get(k, default)

# masks.py
from typing import Tuple, Optional, Dict
import numpy as np
import numpy.typing as npt

def normalize_mask(
    mask: npt.NDArray,
    *,
    mode: str = "percentile",           # "percentile" | "max" | "unit_sum" | "unit_energy"
    percentile: float = 99.0,           # used when mode="percentile"
    subtract_min: bool = True,          # shift so min→0 before scaling
    out_range: Tuple[float, float] = (0.0, 1.0),  # (lo, hi) e.g. (0,1) or (0,255)
    clip: bool = True,                  # clip negatives and upper tail after scaling
    eps: float = 1e-6,
    return_stats: bool = False,
) -> npt.NDArray:
    """
    Normalize a mask to a consistent scale.

    Modes
    -----
    - "percentile": divide by p{percentile} (robust to outliers).  <-- default
    - "max":        divide by max (exact peak=1 but sensitive to single hot pixels).
    - "unit_sum":   divide by sum(|mask|) so total "energy"/power is 1 (size-invariant).
    - "unit_energy":divide by L2 norm (sqrt(sum(mask^2))) for RMS-based scaling.

    Typical usage
    -------------
    Lhat = normalize_mask(L)                     # shape-only in [0,1]
    out  = img + k * Lhat                        # additive later
    out  = img * (1 + k * Lhat)                  # multiplicative later (in linear light)
    """
    m = mask.astype(np.float32, copy=True)

    # Optional shift so baseline is zero (useful if lights only add)
    if subtract_min:
        m -= float(np.min(m))

    # Choose scale
    if mode == "percentile":
        scale = float(np.percentile(m, percentile))
    elif mode == "max":
        scale = float(np.max(m))
    elif mode == "unit_sum":
        scale = float(np.sum(np.abs(m)))
    elif mode == "unit_energy":
        scale = float(np.sqrt(np.sum(m * m)))
    else:
        raise ValueError("normalize_mask: unknown mode %r" % mode)

    scale = max(scale, eps)
    m /= scale

    # Map to output range & clip
    lo, hi = float(out_range[0]), float(out_range[1])
    if clip:
        m = np.clip(m, 0.0, 1.0)
    if not (lo == 0.0 and hi == 1.0):
        m = lo + (hi - lo) * m

    if return_stats:
        return m, {"scale": scale, "min_in": float(np.min(mask)), "max_in": float(np.max(mask))}
    return m


def apply_mask(
    img,
    mask,
    *,
    exposure=1.0,
    max_brightness=255.0,
    high_percentile=99.0,
    margin_high=0.0,
    min_brightness=0.0,
    low_percentile=1.0,
    margin_low=0.0,
    s_min=0.0,
    s_max=10.0,
    clip=True,
    debug=False
):
    """
    Apply an additive mask to an image with optional exposure scaling and
    percentile-based normalization.

    Parameters
    ----------
    img : ndarray
        Input image, shape (H, W) for grayscale or (H, W, C) for multi-channel.
        Can be uint8 or float32.
    mask : ndarray
        Additive mask, same shape as image (or (H, W) for channel-independent mask).
    exposure : float, optional
        Exposure scaling factor applied to the base image before adding the mask.
    max_brightness : float, optional
        Brightness ceiling after adding the mask (applies to high-percentile normalization).
    high_percentile : float, optional
        Percentile of the brightness channel used for ceiling scaling (0–100).
    margin_high : float, optional
        Safety margin subtracted from `max_brightness` when enforcing the ceiling.
    min_brightness : float, optional
        Brightness floor after adding the mask (applies to low-percentile normalization).
    low_percentile : float, optional
        Percentile of the brightness channel used for floor scaling (0–100).
    margin_low : float, optional
        Safety margin added to `min_brightness` when enforcing the floor.
    s_min : float, optional
        Minimum allowed mask scale factor.
    s_max : float, optional
        Maximum allowed mask scale factor.
    clip : bool, optional
        If True, clip the final image values to [min_brightness, max_brightness].
    debug : bool, optional
        If True, prints debug information about percentile values and chosen scale.

    Returns
    -------
    tuple
        processed_img : ndarray
            Image after exposure, mask scaling, and normalization (uint8).
        stats : dict
            Debug dictionary with keys:
            - 's' : chosen mask scale factor
            - 'reason' : reason for chosen scale ('both-bounds', 'ceiling-only', etc.)
            - 'B_hi', 'M_hi' : brightness/mask values at high percentile
            - 'B_lo', 'M_lo' : brightness/mask values at low percentile
            - 's_upper', 's_lower' : scale bounds from ceiling/floor constraints

    Notes
    -----
    - Skips high-percentile calculation if the mask is purely non-positive.
    - Skips low-percentile calculation if the mask is purely non-negative.
    - If both floor and ceiling constraints conflict, chooses the bound with the
      smaller absolute scale magnitude.
    """
    img_f  = img.astype(np.float32, copy=False)
    mask_f = mask.astype(np.float32, copy=False)

    base = exposure * img_f
    gray_base = get_brightness_channel(base, debug=debug)

    # High-percentile ceiling (skip if mask ≤ 0 everywhere)
    check_high = mask_f.max() > 0
    if check_high:
        B_hi = float(np.percentile(gray_base, high_percentile))
        M_hi = float(np.percentile(mask_f,   high_percentile)) + 1e-8
        s_upper = (max_brightness - margin_high - B_hi) / M_hi
    else:
        B_hi = M_hi = None
        s_upper = np.inf

    # Low-percentile floor (skip if mask ≥ 0 everywhere)
    check_low = mask_f.min() < 0
    if check_low:
        B_lo = float(np.percentile(gray_base, low_percentile))
        M_lo = float(np.percentile(mask_f,    low_percentile)) - 1e-8  # likely negative
        # Enforce: B_lo + s*M_lo ≥ min_brightness + margin_low
        s_lower = (min_brightness + margin_low - B_lo) / M_lo
    else:
        B_lo = M_lo = None
        s_lower = -np.inf

    # --- Clamp with user s_min/s_max, and resolve conflicts ---
    s_feasible_low  = max(s_lower, s_min)   # smallest s we’re allowed to use
    s_feasible_high = min(s_upper, s_max)   # largest s we’re allowed to use

    if s_feasible_low <= s_feasible_high:
        # There is an overlap (a feasible interval). Pick the midpoint (neutral choice).
        s = 0.5 * (s_feasible_low + s_feasible_high)
        reason = "both-bounds" if (check_high and check_low) else \
                 ("ceiling-only" if check_high else "floor-only")
    else:
        # No overlap: the floor and ceiling constraints conflict.
        # Choose the nearest boundary to 0 (or, more precisely, the one with smaller magnitude)
        # so we violate the other constraint as little as possible.
        if abs(s_feasible_low) < abs(s_feasible_high):
            s, reason = s_feasible_low,  "floor-priority (conflict)"
        else:
            s, reason = s_feasible_high, "ceiling-priority (conflict)"

    add = s * mask_f
    if img_f.ndim == 3 and img_f.shape[2] != 1:
        add = add[..., None]
    out = base + add

    if clip:
        low_clip = min_brightness if (min_brightness is not None) else 0.0
        out = np.clip(out, low_clip, max_brightness)

    if debug:
        print(f"Exposure={exposure:.3f}, s={s:.4f} ({reason})")
        if check_high:
            print(f"High {high_percentile}%: B_hi={B_hi:.2f}, M_hi≈{M_hi:.2f}, s_upper={s_upper:.4f}")
        if check_low:
            print(f"Low  {low_percentile}%:  B_lo={B_lo:.2f}, M_lo≈{M_lo:.2f}, s_lower={s_lower:.4f}")

    
    out_img = out.astype(np.uint8)
    out_log = {
        "s": float(s),
        "reason": reason,
        "B_hi": B_hi, "M_hi": M_hi,
        "B_lo": B_lo, "M_lo": M_lo,
        "s_upper": float(s_upper),
        "s_lower": float(s_lower),
    }

    return out_img, out_log