# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0] — 2025-08-12
### Added
- **Light generation**
  - `illumimod.masks.generate_light(...)`
  - `illumimod.masks.combine(...)`
  - `illumimod.masks.scale(...)`
  - `illumimod.masks.autocap(...)`
  - `illumimod.masks.apply(...)`

- **Sampling**
  - `illumimod.sampling.sample_truncated_normal(...)`
  - `illumimod.sampling.sample_pixel_index(...)`

- **Examples**
  - `examples/01_demo_single_light`: single light to mask generation.
  - `examples/02_demo_combine masks`: combine multiple light masks.
  - `examples/03_demo_simple_apply`: apply single mask, unnormalized, to image.
  - `examples/04_demo_random_batch`: generate batch of 100 controlled-random lighting variations of same image. Test run for ML-based activities.

- **Tests**
  - `tests/...`: Simple assertions pertaining to above features

- **Packaging**
  - `pyproject.toml` with `src/` layout; optional extras: `illumimod[dev]`, `illumimod[opencv]`.
  - `src/illumimod/__init__.py` (public API surface kept small and stable).

### Notes
- Python ≥ 3.9.
- OpenCV is **optional**; core features are NumPy-only.
- API is intentionally minimal; smart scaling and multiplicative/affine modes will come in later versions without breaking current methods.

