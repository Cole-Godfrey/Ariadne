"""Render loss and accuracy curves as a dependency-free SVG."""

from __future__ import annotations

from html import escape
from pathlib import Path


def _polyline(
    values: list[float], left: int, top: int, width: int, height: int, low: float, high: float
) -> str:
    scale = high - low if high > low else 1.0
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = left + width * index / max(1, count - 1)
        y = top + height * (high - value) / scale
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        raise ValueError("history is empty")
    width, height = 1040, 440
    panels = (("Cross-entropy loss", "loss"), ("Accuracy", "accuracy"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="MNIST training and validation curves">',
        '<rect width="1040" height="440" fill="#fff"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#243144} .title{font-size:18px;font-weight:600} .tick{font-size:11px;fill:#65738a} .legend{font-size:12px}</style>',
    ]
    for panel_index, (title, key) in enumerate(panels):
        left = 68 + panel_index * 510
        top, chart_width, chart_height = 65, 420, 290
        training = [float(row[f"train_{key}"]) for row in history]
        validation = [float(row[f"val_{key}"]) for row in history]
        minimum = 0.0 if key == "loss" else min(0.8, min(training + validation))
        maximum = max(training + validation) * 1.05 if key == "loss" else 1.0
        parts.append(f'<text class="title" x="{left}" y="32">{escape(title)}</text>')
        for tick in range(6):
            y = top + chart_height * tick / 5
            value = maximum - (maximum - minimum) * tick / 5
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e6ebf2"/>')
            parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#718096"/>')
        parts.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#718096"/>')
        for index in range(len(history)):
            if index == 0 or index == len(history) - 1 or (index + 1) % 5 == 0:
                x = left + chart_width * index / max(1, len(history) - 1)
                parts.append(f'<text class="tick" x="{x:.1f}" y="{top + chart_height + 20}" text-anchor="middle">{index + 1}</text>')
        for values, color, label in ((training, "#2463b4", "Training"), (validation, "#dc5e3b", "Validation")):
            points = _polyline(values, left, top, chart_width, chart_height, minimum, maximum)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
            legend_x = left + (0 if label == "Training" else 125)
            parts.append(f'<line x1="{legend_x}" y1="402" x2="{legend_x + 20}" y2="402" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="legend" x="{legend_x + 27}" y="406">{label}</text>')
        parts.append(f'<text class="tick" x="{left + chart_width / 2}" y="390" text-anchor="middle">Epoch</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
