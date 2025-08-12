from typing import Tuple
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
    Create a flat-top, directional radial light mask with an exact half-life from the core edge.

    The footprint is circular at normal incidence and becomes an ellipse for grazing angles
    (via a stretch factor of ``1/sin(angle_deg)`` along the approach direction). Intensity is:

    - ``I = peak`` for ``r_eff <= core_radius`` (flat core)
    - ``I = peak * 2^{-((r_eff - core_radius)/half_life_radius)^power}`` otherwise

    so that ``I(core_radius + half_life_radius) = peak / 2`` by construction.

    Parameters
    ----------
    width : int
        Output width in pixels.
    height : int
        Output height in pixels.
    center : tuple of float
        Light center in pixel coordinates ``(x, y)``.
    peak : float
        Intensity at/inside the flat core (in your image units, e.g., 0–255).
    core_radius : float
        Radius of the flat core in pixels.
    half_life_radius : float
        Distance (in pixels) beyond the core edge at which intensity halves.
    power : float, optional
        Tail steepness. ``1`` = softer/longer tail, ``~2`` ≈ Gaussian-ish, ``>2`` sharper falloff.
    edge_softness : float, optional
        Smooth the core edge over this many pixels using a smoothstep blend. ``0`` = hard edge.
    angle_deg : float, optional
        Incidence angle w.r.t. the surface plane. ``90`` = perpendicular (no stretch),
        smaller values = grazing (more elliptical stretch).
    direction_deg : float, optional
        Approach direction in the image plane, measured **from North (up), clockwise**:
        ``0``=from up, ``90``=from right, ``180``=from down, ``270``=from left.
        At ``angle_deg=90``, direction has no visual effect.
    lambertian : bool, optional
        If ``True``, scale the peak by ``sin(angle_deg)`` (Lambert-like foreshortening).
    dtype : numpy dtype, optional
        Output dtype. Defaults to ``np.float32``.

    Returns
    -------
    ndarray
        Array of shape ``(height, width)`` with dtype ``dtype``.

    Notes
    -----
    - Image coordinates follow NumPy conventions: x→right, y→down.
    - The effective radius uses an **elliptical metric** aligned to ``direction_deg``.
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