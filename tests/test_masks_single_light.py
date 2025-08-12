import numpy as np
import pytest

from illumimod.masks import generate_light as light_to_mask

def test_core_peak_and_half_life_axis_aligned():
    W = H = 161
    cx = cy = 80
    peak = 200.0
    core = 10.0
    half = 20.0
    # 90° (perpendicular), direction irrelevant when angle=90
    m = light_to_mask(
        width=W, height=H, center=(cx, cy),
        peak=peak, core_radius=core, half_life_radius=half,
        power=2.0, edge_softness=0.0, angle_deg=90.0, direction_deg=0.0,
        lambertian=False, dtype=np.float32
    )

    # Inside core -> exactly peak
    assert np.isclose(m[cy, cx], peak, atol=1e-5)

    # At r = core + half along a principal axis -> exactly peak/2
    x_half = int(cx + core + half)   # (80 + 10 + 20) = 110
    assert np.isclose(m[cy, x_half], peak * 0.5, atol=1e-4)

def test_directional_elongation_with_grazing_angle():
    W = H = 201
    cx = cy = 100
    peak = 100.0
    core = 0.0
    half = 40.0
    # Shallow (grazing-ish) angle stretches footprint along approach direction.
    # direction=90° means light comes from the RIGHT; parallel axis = +x
    m = light_to_mask(
        width=W, height=H, center=(cx, cy),
        peak=peak, core_radius=core, half_life_radius=half,
        power=2.0, edge_softness=0.0, angle_deg=30.0, direction_deg=90.0,
        lambertian=False, dtype=np.float32
    )

    # Sample equal offsets along parallel (+x) vs perpendicular (+y).
    dx = 60  # beyond half-life so differences are pronounced
    par_val  = m[cy, cx + dx]    # along approach direction
    perp_val = m[cy + dx, cx]    # perpendicular to approach
    # Expect more energy along the stretched (parallel) axis
    assert par_val > perp_val

def test_additive_application_grayscale():
    W = H = 129
    cx = cy = 64
    img = np.full((H, W), 100, np.uint8)

    peak = 180.0
    core = 8.0
    half = 24.0

    m = light_to_mask(
        width=W, height=H, center=(cx, cy),
        peak=peak, core_radius=core, half_life_radius=half,
        power=2.0, edge_softness=0.0, angle_deg=90.0, direction_deg=0.0,
        lambertian=False, dtype=np.float32
    )

    out = np.clip(img.astype(np.float32) + m, 0, 255).astype(np.uint8)

    # Center should brighten by ~peak, clipped to 255 if needed.
    expected_center = min(255, 100 + int(round(peak)))
    assert out[cy, cx] == expected_center

    # A far-away pixel should be unchanged (mask ~ 0 there).
    assert out[0, 0] == img[0, 0]
