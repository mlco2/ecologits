"""Generate the supported models page from models.json."""

import json
from pathlib import Path

import mkdocs_gen_files

DATA_PATH = Path("ecologits/data/models.json")
OUT_PATH = Path("models.md")


def _fmt_params(arch: dict) -> tuple[str, float]:
    """Return (display string, sort value) for architecture parameters."""
    params = arch.get("parameters")
    if isinstance(params, dict) and "total" in params:
        total = params["total"]
        active = params["active"]
        sort_val = float(active) if isinstance(active, (int, float)) else float(active.get("min", 0))
        t_str = f"{total}B" if isinstance(total, (int, float)) else f"{total['min']}–{total['max']}B"
        a_str = f"{active}B" if isinstance(active, (int, float)) else f"{active['min']}–{active['max']}B"
        return f"{t_str} total / {a_str} active", sort_val
    if isinstance(params, dict) and "min" in params:
        return f"{params['min']}–{params['max']}B", float(params["min"])
    val = float(params)
    return f"{val}B", val


def _fmt_flops(val) -> str:
    """Format raw FLOPs to human-readable TFLOPs."""
    if val is None:
        return "—"
    if isinstance(val, dict):
        lo = val.get("min", 0) / 1e12
        hi = val.get("max", 0) / 1e12
        return f"{lo:.0f}–{hi:.0f} TF"
    return f"{val / 1e12:.0f} TF"


def _warning_badges(warnings) -> str:
    if not warnings:
        return ""
    short = {
        "model-arch-not-released": "closed",
        "model-arch-multimodal": "multimodal",
        "modality-editing-unvalidated": "edit≈",
    }
    badges = []
    for w in warnings:
        label = short.get(w, w.split("-")[-1])
        badges.append(f'<span class="model-warning-badge">{label}</span>')
    return " ".join(badges)


def _row(model: dict) -> str:
    provider = model.get("provider", "")
    name = model.get("name", "")
    modality = model.get("modality", "text")
    arch = model.get("architecture", {})
    arch_type = arch.get("type", "dense")
    param_str, param_sort = _fmt_params(arch)
    diffusion = model.get("diffusion") or {}
    steps = diffusion.get("default_steps", "")
    guidance = diffusion.get("default_guidance_scale", "")
    flops_str = _fmt_flops(diffusion.get("flops_denoise_per_step")) if diffusion else "—"
    warnings = model.get("warnings") or []
    warn_html = _warning_badges(warnings)

    modality_label = {"text": "Text", "image": "Image", "video": "Video"}.get(modality, modality)
    steps_cell = str(steps) if steps != "" else "—"
    guidance_cell = str(guidance) if guidance != "" else "—"
    arch_label = "MoE" if arch_type == "moe" else "Dense"

    return (
        f'<tr data-modality="{modality}">'
        f"<td>{provider}</td>"
        f"<td><code>{name}</code></td>"
        f'<td><span class="modality-badge modality-{modality}">{modality_label}</span></td>'
        f"<td>{arch_label}</td>"
        f'<td data-sort-value="{param_sort}">{param_str}</td>'
        f'<td data-sort-value="{steps if steps != "" else -1}">{steps_cell}</td>'
        f'<td data-sort-value="{guidance if guidance != "" else -1}">{guidance_cell}</td>'
        f"<td>{flops_str}</td>"
        f"<td>{warn_html}</td>"
        "</tr>"
    )


def _build_page(models: list[dict]) -> str:
    counts = {m: sum(1 for x in models if x.get("modality") == m) for m in ("text", "image", "video")}
    total = len(models)

    rows_html = "\n".join(_row(m) for m in models)

    return f"""# Supported Models

All **{total}** models currently tracked in EcoLogits, including LLMs and generative image/video models.
Use the filter buttons to narrow by output modality and click any column header to sort.

<div class="models-filter-bar" markdown="0">
  <span class="filter-label">Modality</span>
  <button class="modality-filter active" data-filter="all">All <span class="filter-count">{total}</span></button>
  <button class="modality-filter" data-filter="text">Text <span class="filter-count">{counts['text']}</span></button>
  <button class="modality-filter" data-filter="image">Image <span class="filter-count">{counts['image']}</span></button>
  <button class="modality-filter" data-filter="video">Video <span class="filter-count">{counts['video']}</span></button>
</div>

<div class="models-search-bar" markdown="0">
  <input id="models-search" type="search" placeholder="Search model name or provider…" autocomplete="off">
  <span id="models-visible-count" class="filter-count-display">{total} models</span>
</div>

<div markdown="0" style="overflow-x:auto">
<table id="models-table">
<thead>
<tr>
  <th>Provider</th>
  <th>Model</th>
  <th>Modality</th>
  <th>Architecture</th>
  <th>Parameters</th>
  <th>Steps</th>
  <th>Guidance Scale</th>
  <th>Denoiser FLOPs/step</th>
  <th>Flags</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
"""


with open(DATA_PATH) as f:
    data = json.load(f)

models = [m for m in data["models"] if m.get("type") == "model"]

with mkdocs_gen_files.open(OUT_PATH, "w") as f:
    f.write(_build_page(models))
