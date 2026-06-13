#!/usr/bin/env python3
"""Generate reproducible SVG charts for the project poster."""

import csv
import math
from html import escape
from pathlib import Path


DATA_DIR = Path("paper_materials")
OUTPUT_DIR = Path("poster")

NAVY = "#1a3a5c"
GOLD = "#c89518"
GREEN = "#2e7d32"
RED = "#b04020"
BLUE = "#4a7a9a"
LIGHT_BLUE = "#eef3f8"
GRID = "#c8d2dc"
TEXT = "#263238"
MUTED = "#607080"
WHITE = "#ffffff"


def read_csv(path, required):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(required) - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def svg_document(width, height, title, content):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
  <title id="title">{escape(title)}</title>
  <rect width="{width}" height="{height}" rx="18" fill="{WHITE}"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: {TEXT}; }}
    .title {{ font-size: 34px; font-weight: 700; fill: {NAVY}; }}
    .subtitle {{ font-size: 18px; fill: {MUTED}; }}
    .axis {{ font-size: 17px; fill: {MUTED}; }}
    .label {{ font-size: 19px; font-weight: 600; }}
    .value {{ font-size: 18px; font-weight: 700; }}
    .note {{ font-size: 16px; fill: {MUTED}; }}
  </style>
{content}
</svg>
"""


def text(x, y, value, css="", anchor="start", fill=None):
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"']
    if css:
        attrs.append(f'class="{css}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    return f'  <text {" ".join(attrs)}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, stroke=GRID, width=2, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def write_svg(name, width, height, title, content):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(svg_document(width, height, title, "\n".join(content)), encoding="utf-8")
    print(f"wrote {path}")


def generate_experiment_trends(rows):
    width, height = 1200, 680
    left, right, top, bottom = 115, 70, 125, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = 60
    x_step = plot_w / (len(rows) - 1)
    points = [left + i * x_step for i in range(len(rows))]
    win = [float(row["win_rate_vs_base_card"]) * 100 for row in rows]
    repetition = [float(row["repetition"]) * 100 for row in rows]

    out = [
        text(55, 55, "Experiment Trend: Win Rate vs. Repetition", "title"),
        text(55, 88, "Single-turn LoRA results across train_1 to train_4", "subtitle"),
    ]
    for tick in range(0, max_y + 1, 10):
        y = top + plot_h - tick / max_y * plot_h
        out.extend(
            [
                line(left, y, width - right, y, GRID, 1),
                text(left - 18, y + 6, f"{tick}%", "axis", "end"),
            ]
        )
    out.extend(
        [
            line(left, top, left, top + plot_h, NAVY, 2),
            line(left, top + plot_h, width - right, top + plot_h, NAVY, 2),
        ]
    )

    for x, row in zip(points, rows):
        out.append(text(x, top + plot_h + 42, row["experiment"], "label", "middle"))

    for values, color in ((win, GREEN), (repetition, RED)):
        coords = [
            f"{x:.1f},{top + plot_h - value / max_y * plot_h:.1f}"
            for x, value in zip(points, values)
        ]
        out.append(
            f'  <polyline points="{" ".join(coords)}" fill="none" stroke="{color}" '
            'stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, value in zip(points, values):
            y = top + plot_h - value / max_y * plot_h
            out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="{WHITE}" stroke-width="3"/>')
            offset = -18 if color == GREEN else 31
            out.append(text(x, y + offset, f"{value:.1f}%", "value", "middle", color))

    out.extend(
        [
            f'  <line x1="680" y1="61" x2="735" y2="61" stroke="{GREEN}" stroke-width="6"/>',
            text(750, 68, "Win rate vs. Base + card", "label"),
            f'  <line x1="950" y1="61" x2="1005" y2="61" stroke="{RED}" stroke-width="6"/>',
            text(1020, 68, "Repetition", "label"),
            text(55, 650, "Higher win rate is better; lower repetition is better.", "note"),
        ]
    )
    write_svg("experiment-trends.svg", width, height, "Experiment trends", out)


def radar_chart(name, title_value, subtitle, labels, series):
    width, height = 900, 760
    cx, cy, radius = 450, 405, 245
    angles = [-math.pi / 2 + i * 2 * math.pi / len(labels) for i in range(len(labels))]
    out = [
        text(45, 54, title_value, "title"),
        text(45, 86, subtitle, "subtitle"),
    ]

    for level in range(1, 6):
        r = radius * level / 5
        pts = " ".join(
            f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}"
            for angle in angles
        )
        out.append(
            f'  <polygon points="{pts}" fill="{LIGHT_BLUE if level % 2 else WHITE}" '
            f'fill-opacity="0.55" stroke="{GRID}" stroke-width="1.5"/>'
        )
        out.append(text(cx + 8, cy - r + 18, level, "axis"))

    for label, angle in zip(labels, angles):
        x2 = cx + radius * math.cos(angle)
        y2 = cy + radius * math.sin(angle)
        out.append(line(cx, cy, x2, y2, GRID, 1.5))
        label_r = radius + 54
        x = cx + label_r * math.cos(angle)
        y = cy + label_r * math.sin(angle) + 7
        anchor = "middle"
        if math.cos(angle) > 0.35:
            anchor = "start"
        elif math.cos(angle) < -0.35:
            anchor = "end"
        out.append(text(x, y, label, "label", anchor))

    for index, (series_name, values, color) in enumerate(series):
        pts = []
        for value, angle in zip(values, angles):
            r = radius * value / 5
            pts.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
        out.append(
            f'  <polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.18" '
            f'stroke="{color}" stroke-width="5" stroke-linejoin="round"/>'
        )
        for value, angle in zip(values, angles):
            r = radius * value / 5
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        legend_x = 185 + index * 310
        out.append(f'  <rect x="{legend_x}" y="710" width="34" height="12" rx="6" fill="{color}"/>')
        out.append(text(legend_x + 48, 724, series_name, "label"))

    write_svg(name, width, height, title_value, out)


def generate_radars(rows):
    systems = {row["system"]: row for row in rows}
    base = systems["base_with_card"]
    lora = systems["lora_with_card"]

    radar_chart(
        "train4-single-radar.svg",
        "train_4 Single-Turn Judge",
        "Five dimensions scored from 1 to 5",
        ["Identity", "Style", "Relevance", "Naturalness", "Immersion"],
        [
            (
                "Base + card",
                [float(base[key]) for key in (
                    "single_identity", "single_style", "single_relevance",
                    "single_naturalness", "single_immersion",
                )],
                GOLD,
            ),
            (
                "LoRA + card",
                [float(lora[key]) for key in (
                    "single_identity", "single_style", "single_relevance",
                    "single_naturalness", "single_immersion",
                )],
                GREEN,
            ),
        ],
    )

    radar_chart(
        "train4-multi-radar.svg",
        "train_4 Four-Turn Challenge",
        "LoRA only matches the prompt baseline on memory",
        ["Identity", "Memory", "Coherence", "Style", "Immersion"],
        [
            (
                "Base + card",
                [float(base[key]) for key in (
                    "multi_identity", "multi_memory", "multi_coherence",
                    "multi_style", "multi_immersion",
                )],
                GOLD,
            ),
            (
                "LoRA + card",
                [float(lora[key]) for key in (
                    "multi_identity", "multi_memory", "multi_coherence",
                    "multi_style", "multi_immersion",
                )],
                GREEN,
            ),
        ],
    )


def generate_character_card_impact(rows):
    width, height = 1200, 720
    left, right, top, bottom = 145, 70, 125, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    systems = [
        ("base_no_card", "Base, no card", NAVY),
        ("base_with_card", "Base + card", GOLD),
        ("lora_with_card", "LoRA + card", GREEN),
    ]
    data = {row["system"]: row for row in rows}
    group_step = plot_w / len(systems)
    bar_width = 92
    out = [
        text(55, 55, "Character Card Contribution in train_4", "title"),
        text(55, 88, "Weighted judge score: prompt benefit is larger than LoRA's extra gain", "subtitle"),
    ]
    for tick in range(0, 6):
        y = top + plot_h - tick / 5 * plot_h
        out.extend(
            [
                line(left, y, width - right, y, GRID, 1),
                text(left - 18, y + 6, tick, "axis", "end"),
            ]
        )
    out.extend(
        [
            line(left, top, left, top + plot_h, NAVY, 2),
            line(left, top + plot_h, width - right, top + plot_h, NAVY, 2),
        ]
    )

    for i, (key, label, color) in enumerate(systems):
        center = left + group_step * (i + 0.5)
        values = [
            ("Single-turn", float(data[key]["single_weighted"])),
            ("Four-turn", float(data[key]["multi_weighted"])),
        ]
        for j, (metric, value) in enumerate(values):
            x = center + (j - 0.5) * (bar_width + 22) - bar_width / 2
            bar_h = value / 5 * plot_h
            y = top + plot_h - bar_h
            opacity = 1 if j == 0 else 0.55
            out.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" '
                f'rx="8" fill="{color}" fill-opacity="{opacity}"/>'
            )
            out.append(text(x + bar_width / 2, y - 12, f"{value:.3f}", "value", "middle", color))
        out.append(text(center, top + plot_h + 45, label, "label", "middle"))

    out.extend(
        [
            f'  <rect x="410" y="665" width="32" height="18" rx="4" fill="{BLUE}"/>',
            text(455, 680, "Single-turn", "label"),
            f'  <rect x="650" y="665" width="32" height="18" rx="4" fill="{BLUE}" fill-opacity="0.55"/>',
            text(695, 680, "Four-turn", "label"),
            text(640, 165, "+0.765 prompt gain", "value", "middle", GOLD),
            text(915, 165, "+0.235 LoRA gain", "value", "middle", GREEN),
        ]
    )
    write_svg("character-card-impact.svg", width, height, "Character card contribution", out)


def generate_paired_effects(rows):
    width, height = 1200, 760
    left, right, top, bottom = 220, 90, 125, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -1.8, 0.8

    def sx(value):
        return left + (value - x_min) / (x_max - x_min) * plot_w

    out = [
        text(55, 55, "Paired Effect: LoRA - Base + Card", "title"),
        text(55, 88, "Weighted-score difference with bootstrap 95% confidence intervals", "subtitle"),
    ]
    for tick in (-1.5, -1.0, -0.5, 0.0, 0.5):
        x = sx(tick)
        out.extend(
            [
                line(x, top, x, top + plot_h, RED if tick == 0 else GRID, 3 if tick == 0 else 1),
                text(x, top + plot_h + 35, f"{tick:+.1f}", "axis", "middle"),
            ]
        )

    row_step = plot_h / len(rows)
    for i, row in enumerate(rows):
        y = top + row_step * (i + 0.5)
        out.append(text(left - 28, y + 7, row["experiment"], "label", "end"))
        out.append(line(left, y + row_step / 2, width - right, y + row_step / 2, GRID, 1))

        single = float(row["single_diff_lora_minus_base_card"])
        low = float(row["single_ci95_low"])
        high = float(row["single_ci95_high"])
        multi = float(row["multi_diff_lora_minus_base_card"])
        out.extend(
            [
                line(sx(low), y - 17, sx(high), y - 17, GREEN, 5),
                line(sx(low), y - 27, sx(low), y - 7, GREEN, 3),
                line(sx(high), y - 27, sx(high), y - 7, GREEN, 3),
                f'  <circle cx="{sx(single):.1f}" cy="{y - 17:.1f}" r="9" fill="{GREEN}"/>',
                f'  <rect x="{sx(multi) - 8:.1f}" y="{y + 8:.1f}" width="16" height="16" fill="{RED}"/>',
                text(width - right + 8, y + 7, f"win {float(row['single_win_rate']) * 100:.1f}%", "value", "start", GREEN),
            ]
        )

    out.extend(
        [
            f'  <line x1="315" y1="710" x2="365" y2="710" stroke="{GREEN}" stroke-width="5"/>',
            f'  <circle cx="340" cy="710" r="8" fill="{GREEN}"/>',
            text(380, 717, "Single-turn mean and 95% CI", "label"),
            f'  <rect x="705" y="702" width="16" height="16" fill="{RED}"/>',
            text(735, 717, "Four-turn mean", "label"),
            text(55, 745, "All single-turn intervals cross zero; all four-turn means are negative.", "note"),
        ]
    )
    write_svg("paired-effects.svg", width, height, "Paired effects", out)


def main():
    experiments = read_csv(
        DATA_DIR / "experiment_comparison.csv",
        {"experiment", "win_rate_vs_base_card", "repetition"},
    )
    paired = read_csv(
        DATA_DIR / "paired_comparison.csv",
        {
            "experiment", "single_diff_lora_minus_base_card", "single_ci95_low",
            "single_ci95_high", "single_win_rate", "multi_diff_lora_minus_base_card",
        },
    )
    systems = read_csv(
        DATA_DIR / "train4_system_comparison.csv",
        {
            "system", "single_identity", "single_style", "single_relevance",
            "single_naturalness", "single_immersion", "single_weighted",
            "multi_identity", "multi_memory", "multi_coherence", "multi_style",
            "multi_immersion", "multi_weighted",
        },
    )

    generate_experiment_trends(experiments)
    generate_radars(systems)
    generate_character_card_impact(systems)
    generate_paired_effects(paired)


if __name__ == "__main__":
    main()
