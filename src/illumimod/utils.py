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

