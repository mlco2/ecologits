"""
Diffusion model impacts — worked examples.

Demonstrates compute_diffusion_impacts() for:
  1. SDXL text-to-image (1024 × 1024, 40 steps, CFG=7.5)
  2. SDXL image-to-image editing (denoising_strength=0.6)
  3. CogVideoX-5b text-to-video (49 frames, 50 steps)
  4. Flux.1-schnell text-to-image (flow-matching, no CFG, 4 steps)

All energy figures are returned as RangeValue intervals reflecting:
  - GPU efficiency calibration uncertainty (2.5 – 5.0 × 10⁻¹⁸ kWh/FLOP)
  - Proprietary-model FLOPs uncertainty (where applicable)
"""

from ecologits.tracers.utils import diffusion_impacts


def fmt(label: str, out) -> None:
    """Print a compact summary of an ImpactsOutput."""
    e = out.energy.value
    gwp = out.gwp.value
    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  Energy  : {e}  kWh")
    print(f"  GWP     : {gwp}  kgCO2eq")
    if out.warnings:
        print(f"  Warnings: {[w.code for w in out.warnings]}")
    if out.errors:
        print(f"  Errors  : {[e.code for e in out.errors]}")


# ── 1. SDXL text-to-image ────────────────────────────────────────────────────
# 40 denoising steps × 2 (CFG double-pass) × 6.7 TFLOPs/step = 536 TFLOPs
# Measured by AI Energy Score on A100: 1.64 Wh — within our lower bound.
out1 = diffusion_impacts(
    provider="stabilityai",
    model_name="stable-diffusion-xl",
    task="txt2img",
    request_latency=4.2,   # wall-clock seconds from your API call
    width=1024,
    height=1024,
)
fmt("SDXL • txt2img • 1024 × 1024 • 40 steps • 4.2 s", out1)

# ── 2. SDXL image-to-image editing ─────────────────────────────────────────
# denoising_strength=0.6 → 40 × 0.6 × 2 = 48 effective forward passes.
# Energy returned as a RangeValue; editing-unvalidated warning is added.
out2 = diffusion_impacts(
    provider="stabilityai",
    model_name="stable-diffusion-xl",
    task="img2img",
    request_latency=2.5,
    denoising_strength=0.6,
)
fmt("SDXL • img2img • denoising_strength=0.6 • 2.5 s", out2)

# ── 3. CogVideoX-5b text-to-video ───────────────────────────────────────────
# 170 TFLOPs/step × 50 steps × 2 (CFG) = 17 000 TFLOPs for 49 default frames.
# CogVideoX-5b has 5B parameters → requires 2 × H100-80 GB GPUs.
out3 = diffusion_impacts(
    provider="huggingface_hub",
    model_name="cogvideox-5b",
    task="txt2vid",
    request_latency=90.0,
    n_frames=49,
)
fmt("CogVideoX-5b • txt2vid • 49 frames • 50 steps • 90 s", out3)

# ── 4. Flux.1-schnell text-to-image ─────────────────────────────────────────
# Flow-matching, no CFG double-pass, only 4 steps → 4 × 98 TFLOPs = 392 TFLOPs.
# Despite similar per-step cost to large video models, the tiny step count
# makes it the most efficient image model in the catalogue.
out4 = diffusion_impacts(
    provider="black_forest_labs",
    model_name="flux.1-schnell",
    task="txt2img",
    request_latency=1.2,
    width=1024,
    height=1024,
)
fmt("Flux.1-schnell • txt2img • 1024 × 1024 • 4 steps • 1.2 s", out4)

print(f"\n{'─'*58}")
print("All values include server overhead and PUE.")
print("GPU-only energy ≈ lower end of the Energy range.")
