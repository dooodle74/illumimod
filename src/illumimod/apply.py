import numpy as np

def get_brightness_channel(arr, luma=(0.2126, 0.7152, 0.0722), debug=False):
    """
    Extract a brightness channel from a grayscale or RGB image.

    Parameters
    ----------
    arr : ndarray
        Image array. Can be:
        - 2D array (H, W): interpreted as grayscale.
        - 3D array (H, W, 1): single-channel grayscale, squeezed to (H, W).
        - 3D array (H, W, 3): RGB image, converted to brightness using luma weights.
        - Any other shape: channel-wise mean brightness is computed.
    luma : tuple of float, optional
        Luma weights (R, G, B) for converting RGB to brightness. Defaults to
        Rec. 709 weights: (0.2126, 0.7152, 0.0722).
    debug : bool, optional
        If True, prints debug messages about the branch taken.

    Returns
    -------
    ndarray
        2D array of brightness values, shape (H, W), dtype inherited from input.

    Notes
    -----
    - Luma weighting matches human perception of color brightness.
    - Automatically skips luma weighting if the image is grayscale.

    See Also
    --------
    apply_mask : Uses this method internally to compute brightness percentiles
        for exposure scaling and mask normalization.
    """
    if arr.ndim == 2:
        if debug: print("Brightness: grayscale (H,W).")
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            if debug: print("Brightness: single channel (H,W,1) -> squeeze.")
            return arr[..., 0]
        if arr.shape[2] == 3:
            if debug: print("Brightness: RGB (H,W,3) -> luma weights", luma)
            r, g, b = luma
            return arr[..., 0]*r + arr[..., 1]*g + arr[..., 2]*b
        if debug: print(f"Brightness: {arr.shape[2]} channels -> mean.")
        return arr.mean(axis=2)
    if debug: print(f"Brightness: unexpected shape {arr.shape} -> passthrough.")
    return arr

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