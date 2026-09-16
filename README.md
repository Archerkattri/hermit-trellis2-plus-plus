<div align="center">

<img src="assets/banner.png" alt="hermit-trellis2-plus-plus" width="680">

# 🧭 hermit-trellis2++

<p>
  <a href="https://github.com/Archerkattri/hermit-trellis2-plus-plus/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Archerkattri/hermit-trellis2-plus-plus?color=1f6feb"></a>
  <a href="https://github.com/Archerkattri/hermit-trellis2-plus-plus/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/Archerkattri/hermit-trellis2-plus-plus/total?label=downloads&color=1f6feb"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/Archerkattri/hermit-trellis2-plus-plus?color=0d9488"></a>
</p>


**Training-free acceleration for [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS) image-to-3D — a selectable Hermite/DMD forecast variant of [`hermit-trellis2`](https://github.com/Archerkattri/hermit-trellis2), one line of code.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](./LICENSE)
[![base: TRELLIS.2](https://img.shields.io/badge/base-microsoft%2FTRELLIS.2-555.svg?style=flat-square)](https://github.com/microsoft/TRELLIS.2)
[![arXiv: TRELLIS.2](https://img.shields.io/badge/arXiv-2512.14692%20TRELLIS.2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2512.14692)
[![arXiv: HiCache](https://img.shields.io/badge/arXiv-2508.16984%20HiCache-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.16984)
[![arXiv: Adaptive Guidance](https://img.shields.io/badge/arXiv-2312.12487%20Adaptive%20Guidance-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.12487)

`TRELLIS.2-4B` · `1024_cascade` (mesh + texture) · training-free · single RTX 5090 · MIT

</div>

## Sampler integration

![hermit-trellis2++ sampler integration](assets/readme_flow.svg)

Token carving and velocity forecasting are controlled separately. The selected Hermite or DMD backend acts only inside the TRELLIS.2 flow loop and reports compute, forecast, and fallback steps.

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis. Use it when you want the DMD compatibility path; quality and speed remain workload- and configuration-dependent.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `hermit-trellis2-plus-plus` — **TRELLIS.2-4B × selectable HiCache (Hermite/DMD)** — carved-hybrid. The benchmark card below is historical evidence, not a guarantee for new hardware, checkpoints, or schedules.

`hermit-trellis2++` is `TRELLIS.2-4B` image-to-3D with the same **training-free carved-hybrid**
as [`hermit-trellis2`](https://github.com/Archerkattri/hermit-trellis2) — but with the
sparse-structure velocity forecast on an **exponential Dynamic-Mode-Decomposition (DMD / Prony)
basis** instead of the Hermite polynomial. It forecasts the model's **final CFG-combined velocity**
and **carves** the structured-latent tokens, so the sampler spends far fewer network evaluations per
asset, with the weights, decoders, and the full `1024_cascade` mesh + texture left untouched.

```python
pipe.enable_faster()                                      # carved-hybrid, Hermite SS forecast (default)
pipe.enable_faster("dmd")                                # carved-hybrid, DMD/Prony SS forecast
pipe.enable_faster("base")                                # stock TRELLIS.2 sampler (kill-switch)
```

**What's new vs `hermit-trellis2`.** Same carved-hybrid schedule, same token-carved SLaT stages —
the only change is the **forecast basis on the sparse-structure stage**:

- **HiCache++ (exponential DMD/Prony)** on the **sparse-structure** stage — forecasts the velocity
  with **Dynamic Mode Decomposition** instead of the dual-scaled Hermite polynomial. DMD is an
  exponential forecast basis; its behavior at larger skip intervals is workload-dependent. The early steps,
  where topology is decided, are still always computed.
- **Token-carved SLaT** on the **structured-latent** stages — unchanged from `hermit-trellis2`: a
  learned-cadence temporal skip plus spatial **token carving** that recomputes only the
  high-frequency voxels each step.

**Relationship to the family.** [`hermit-trellis2`](https://github.com/Archerkattri/hermit-trellis2)
is the **HiCache (Hermite)** parent — the v2 instance of the Hermite carved-hybrid (whose v1 sibling
[faster-trellis](https://github.com/Archerkattri/faster-trellis) beats Fast-TRELLIS on both speed and
quality). `hermit-trellis2++` keeps that exact carved-hybrid and swaps the Hermite forecast for the
exponential DMD one. The DMD forecaster ships as a standalone library in
[`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus); this repo is its
TRELLIS.2-v2 integration, selectable with the sampler's `backend="dmd"`.

---

## Quickstart

```bash
git clone https://github.com/Archerkattri/hermit-trellis2-plus-plus
cd hermit-trellis2-plus-plus
# TRELLIS.2 runtime deps (torch, flash-attn, spconv/flex_gemm, o-voxel, cumesh,
# nvdiffrast) per microsoft/TRELLIS.2. Place / symlink weights at ckpts/TRELLIS.2-4B.
```

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image

pipe = Trellis2ImageTo3DPipeline.from_pretrained("ckpts/TRELLIS.2-4B").to("cuda")
pipe.enable_faster("dmd")                                 # carved-hybrid, exponential DMD forecast

out  = pipe.run(Image.open("input_rgba.png"), pipeline_type="1024_cascade")
mesh = out[0]
```

`enable_faster()` defaults to the Hermite forecast for compatibility with `hermit-trellis2`;
`enable_faster("dmd")` selects the exponential forecast. The DMD snapshot-window length is the
sampler's `history` attribute (default `6`). `pipe.acceleration_status()` reports the selected
backend and the independent carved SLaT stages.

`example_faster.py` is the runnable end-to-end script; `example.py` is the stock TRELLIS.2 demo.

<details>
<summary><b>RTX 50-series (sm_120) launch env</b></summary>

`1024_cascade` fits in 32 GB with `expandable_segments`:

```bash
SPARSE_CONV_BACKEND=spconv SPCONV_ALGO=native ATTN_BACKEND=flash_attn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python example_faster.py --image input_rgba.png
```

`SPCONV_ALGO=native` is recommended on newer GPU architectures.
</details>

---

## Historical results (as measured)

The following card is retained as prior-run evidence from the stated benchmark setup. It is not a
current acceptance result and does not establish a universal speedup, losslessness, or quality
ordering. Re-run the manifest command in `example_faster.py` on the target GPU before making a new
claim.

TRELLIS.2-4B, Toys4K, mesh F-score@0.05 (area-weighted surface samples), 40 objects.

**At the deployed interval (`GF_HICACHE_SS_INTERVAL=2`, ~1.9×), matched n=36** (objects that
succeeded for every variant; the rotationally-degenerate sphere excluded):

| backend | F1 mean | F1 median | Chamfer↓ | speedup |
|---|---:|---:|---:|---:|
| accel off (baseline) | 0.902 | 0.956 | 0.044 | 1.00× |
| Fast-TRELLIS.2 (TaylorSeer) | 0.907 | 0.952 | 0.044 | 1.90× |
| HiCache (Hermite) | 0.896 | 0.965 | 0.048 | 1.90× |
| **HiCache++ (DMD)** | **0.900** | 0.960 | **0.047** | 1.89× |

In that historical run, DMD and Hermite were close to the accel-off baseline at the stated schedule;
the values should be interpreted only within that benchmark's matched sample.

**The exponential basis earns its keep as the skip interval grows** (matched n=35, Hermite vs DMD):

| `GF_HICACHE_SS_INTERVAL` | Hermite F1mean | DMD F1mean | Hermite F1med | DMD F1med | DMD − Hermite (mean / med) |
|---|---:|---:|---:|---:|---:|
| 2 (~1.9×, deployed) | 0.894 | 0.900 | 0.969 | 0.962 | +0.005 / −0.007 |
| **3** | 0.836 | **0.872** | 0.930 | **0.946** | **+0.036 / +0.015** |
| **4** | 0.839 | **0.868** | 0.898 | **0.935** | **+0.029 / +0.037** |
| 5 | 0.886 | 0.881 | 0.943 | 0.962 | −0.005 / +0.019 |

**Finding from that run.** At the deployed interval the two bases were close; at intervals 3–4,
DMD scored higher in that matched sample. This does not predict behavior on other checkpoints or
schedules. The standalone microbench and the Hunyuan3D-2.1 i3→i6 sweep in
[`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) predict. (Interval-5 is
non-monotonic — both partially recover, DMD keeping the median lead — an artifact of the carved
schedule's adaptive clamp; reported as measured.) This is a historical TRELLIS.2-4B observation,
not a general claim that the exponential basis extends a near-lossless skip range.

---


### hicache-pp 1.2.0 alignment (2026-06-10)

Two updates relative to [hicache-plus-plus 1.2.0](https://github.com/Archerkattri/hicache-plus-plus):

- **Hermite comparison arm corrected.** The vendored Hermite forecast (the HiCache baseline
  arm, also the DMD warm-up fallback) evaluated the basis at `x = -k`; corrected to `x = +k`
  (the upstream TaylorSeer distance convention; `-k` flips every odd-order term). The
  published numbers above were measured with the as-released code and remain valid
  as-measured. The DMD arm itself is unaffected by the sign convention.
- **Eigencache not yet vendored.** hicache-plus-plus 1.2.0 caches the DMD eigendecomposition
  per compute window; the DMD fit vendored here still refits on every skipped step. That is
  forecast-side latency overhead only (quality is identical); the standalone library ships
  the cached fit, and porting it here is pending.

## How it works

TRELLIS.2 samples a shape in three flow-matching stages — **sparse structure (SS)**, **shape
SLaT** (the 512→1024 cascade), and **texture SLaT** (guidance = 1, no CFG) — each a short Euler
sampler. Both accelerators act on the final velocity `pred_v` those samplers emit.

<details>
<summary><b>① HiCache++ — exponential (DMD/Prony) velocity forecast</b> (replaces network calls on skipped steps)</summary>

At each **compute** step the sampler runs the model and records `pred_v` into a short snapshot
window (the last `history` compute steps). At a **skipped** step it forecasts the velocity by
**Dynamic Mode Decomposition** over those snapshots instead of touching the network:

```
# DMD identifies the linear propagator A from the velocity snapshots,
# its eigen-decomposition A = Φ Λ Φ⁻¹, and advances the modes k steps:
F̂_{t+k} ≈ Φ (Λᵏ · b)
```

**Why exponential over polynomial.** A diffusion feature/velocity trajectory can be modeled by a
near-linear feature-ODE, whose solution class is a sum of (damped / oscillatory)
**exponentials** `Σ bⱼ λⱼᵏ`, not polynomials. DMD — the modern generalisation of Prony's method
(1795) — fits exactly that class: it recovers the modes `(Φ, Λ)` from the snapshots and is **exact on
an exponential series**, which the polynomial Hermite/Taylor bases are not. The polynomial forecasts
grow without bound as the skip horizon `k` increases (Taylor diverges fastest; the dual-scaled
Hermite contracts it but is still polynomial), so the exponential basis is what holds quality at the
larger compute intervals this variant can target, subject to validation. For the dense SS latent `pred_v` is forecast
directly; for the SLaT `SparseTensor`s only `.feats` is forecast and coords carry through via
`.replace(feats)`. With too short a window DMD falls back to reusing the last computed velocity.
*(arXiv:2508.16984 for the HiCache/Hermite parent method; the DMD/Prony basis is the `backend="dmd"`
extension.)*
</details>

<details>
<summary><b>② Token-carved SLaT — recompute only the high-frequency voxels</b> (SLaT stages, unchanged from hermit-trellis2)</summary>

The SLaT stages denoise a `SparseTensor` of voxel tokens, and most tokens change slowly between
steps. On each computed step we score every token by **spatial high-frequency energy** (a 3D-FFT
of the sparse-structure occupancy grid) together with its velocity magnitude and frame-to-frame
motion, and recompute only the most active fraction; the smoothest tokens reuse their cached
velocity, under a staleness bound that forces a periodic full refresh so no token drifts. On top of
that a **learned-k delta cache** skips whole steps when the velocity field is locally linear
(`vₜ ≈ xₜ + Δ`). The SS occupancy's per-token frequency score is the same one the SS forecast stage
reads, so the two stages share one signal. *(Fast-TRELLIS token selection; carving level = `GF_CARVE_RATIO`.)*
</details>

<details>
<summary><b>③ Per-stage split</b> (exponential forecast on SS, token carving on SLaT)</summary>

The two accelerations are matched to what each stage costs. The **sparse-structure** stage is a
small dense volume that fixes the asset's topology — the selected Hermite or DMD forecast thins it
while always computing the first six steps (`GF_HICACHE_FIRST_ENHANCE`), so the occupancy can't be corrupted.
The **shape and texture SLaT** stages are the sparse, expensive ones — token carving recomputes only
their high-frequency voxels per step and the delta cache skips whole steps. The pipeline computes the
SS occupancy's 3D-FFT frequency score once and hands it to the SLaT sampler (`set_coords_scores`),
so the carving signal is the SS structure itself — wired in `trellis2/pipelines/trellis2_image_to_3d.py`.

**The savings multiply:** the SLaT sampler skips whole steps (delta cache) *and* carves tokens on
the steps it does run, while the selected SS forecast independently reduces SS model calls.
</details>

---

## Tuning

One shipped configuration; each knob is overridable by env var (takes precedence) or directly on
the swapped sampler instances. The forecast basis itself is the SS sampler's `hicache_backend`
attribute (`"hermite"` default, `"dmd"` for the exponential variant):

| knob | env | default | meaning |
|---|---|:--:|---|
| carving level | `GF_CARVE_RATIO` | `0.10` | fraction of SLaT tokens cached/skipped per step |
| SS interval | `GF_HICACHE_SS_INTERVAL` | `2` | sparse-structure: compute 1 step, forecast `interval − 1` |
| SS first-enhance | `GF_HICACHE_FIRST_ENHANCE` | `6` | always compute the first N SS steps (protects topology) |

```python
import os; os.environ["GF_CARVE_RATIO"] = "0.15"
pipe.enable_faster()
# …or set the instances directly, after enable_faster():
pipe.sparse_structure_sampler.hicache_backend  = "dmd"   # compatibility override (default "hermite")
pipe.sparse_structure_sampler.hicache_interval = 2
pipe.shape_slat_sampler.carving_ratio          = 0.10
```

---

## What's added on top of TRELLIS.2

All Microsoft TRELLIS.2 model / decoder / o-voxel code is unchanged. Added files only:

- `trellis2/pipelines/samplers/hicache.py` — velocity-forecast cache: Hermite polynomial **and the DMD/Prony exponential** backend (selected by `backend=`), plus the finite-difference / snapshot machinery
- `trellis2/pipelines/samplers/hicache_freq.py` — 3D-FFT high-frequency token scoring (the carving signal)
- `trellis2/pipelines/samplers/flow_euler_carved.py` — the token-carved SLaT sampler (delta-cache step-skip + carving)
- `trellis2/pipelines/samplers/flow_euler.py` — `HiCacheMixin` (`hicache_backend` selector) + the accelerated sampler classes
- `trellis2/pipelines/samplers/__init__.py` — registers the accelerated samplers
- `trellis2/pipelines/trellis2_image_to_3d.py` — `enable_faster()` (single config) + the per-stage wiring
- `example_faster.py`

The accelerators are independent re-implementations of the cited methods on the TRELLIS.2 sampler API.

---

## Credits & license

| | |
|---|---|
| **TRELLIS.2** | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) — the pipeline, models, decoders this builds on (MIT) |
| **hermit-trellis2** | [Archerkattri/hermit-trellis2](https://github.com/Archerkattri/hermit-trellis2) — the HiCache (Hermite) parent this carved-hybrid is built on |
| **hicache-plus-plus** | [Archerkattri/hicache-plus-plus](https://github.com/Archerkattri/hicache-plus-plus) — the standalone DMD/Prony exponential-forecast library |
| **HiCache** | arXiv:2508.16984 — Hermite-polynomial velocity forecasting (the parent method the `++` extends) |
| **DMD / Prony** | Schmid, *Dynamic Mode Decomposition of numerical and experimental data* (JFM 2010); de Prony (1795) — the exponential-forecast basis |
| **Fast-TRELLIS** | [wlfeng0509/Fast-SAM3D (Fast-TRELLIS branch)](https://github.com/wlfeng0509/Fast-SAM3D/tree/Fast-TRELLIS) — the token-carving substrate the SLaT stage builds on |

MIT. Accelerations © 2026 Krishi Attri; bundled TRELLIS.2 © Microsoft Corporation. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**Krishi Attri** · krishiattriwork@gmail.com · [github.com/Archerkattri](https://github.com/Archerkattri)

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@software{attri2026hermittrellis2pp,
  author = {Krishi Attri},
  title  = {hermit-trellis2++: Training-free DMD/Prony exponential-forecast acceleration of TRELLIS.2 image-to-3D},
  year   = {2026},
  url    = {https://github.com/Archerkattri/hermit-trellis2-plus-plus}
}
@article{hicache2025,
  title   = {HiCache: Training-free Acceleration of Diffusion Models via
             Hermite Polynomial Feature Forecasting},
  journal = {arXiv preprint arXiv:2508.16984}, year = {2025}
}
@article{schmid2010dmd,
  title   = {Dynamic mode decomposition of numerical and experimental data},
  author  = {Schmid, Peter J.},
  journal = {Journal of Fluid Mechanics}, volume = {656}, year = {2010}
}
@article{trellis2,
  title   = {Native and Compact Structured Latents for 3D Generation (TRELLIS.2)},
  journal = {arXiv preprint arXiv:2512.14692}, note = {microsoft/TRELLIS.2}
}
```
</details>

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[microsoft/TRELLIS](https://github.com/microsoft/TRELLIS), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.

---

## Family

Part of the **HiCache++ acceleration family**.

- **Family hub:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the basis library behind this adapter.
- **Sibling:** [`hermit-trellis2`](https://github.com/Archerkattri/hermit-trellis2) — the same base model with the HiCache (scaled-Hermite) polynomial-forecast variant.

## Current release status

The current adapter includes shared HiCache++ cache identity, timing and
fallback accounting. Five CPU sampler/contract tests pass. The real sparse
model/CUDA workflow and output-quality comparison remain unmeasured.
