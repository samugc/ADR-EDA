#!/usr/bin/env python3
"""Build thesis-ready RealBike h336 tables and SVG figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICIES = [
    "full_rebalance",
    "rolling_horizon_rebalance",
    "eda_hourly_mo_no_reuse",
    "adr_eda_station_mo",
    "adr_eda_station",
]

LABELS = {
    "full_rebalance": "Reeq. horario",
    "rolling_horizon_rebalance": "Horizonte rodante",
    "eda_hourly_mo_no_reuse": "EDA horaria",
    "adr_eda_station_mo": "ADR-EDA",
    "adr_eda_station": "ADR-EDA (utilidad)",
}

PAIR_LABELS = {
    "full_rebalance": "reeq. horario",
    "rolling_horizon_rebalance": "horizonte rodante",
    "eda_hourly_mo_no_reuse": "EDA horaria",
}

COLORS = {
    "full_rebalance": "#9f1239",
    "rolling_horizon_rebalance": "#0891b2",
    "eda_hourly_mo_no_reuse": "#a21caf",
    "adr_eda_station_mo": "#16a34a",
    "adr_eda_station": "#2563eb",
}


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[row["policy"]] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"policy"} and value not in {"", None}
            }
        return rows


def read_pairwise(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_pair(rows: list[dict[str, str]], lhs: str, rhs: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("lhs") == lhs and row.get("rhs") == rhs:
            return row
    return None


def scale(value: float, lo: float, hi: float, start: float, end: float) -> float:
    if hi == lo:
        return (start + end) / 2
    return start + (value - lo) * (end - start) / (hi - lo)


def axis_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi == lo:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def draw_panel(
    lines: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    rows: dict[str, dict[str, float]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    y_digits: int,
) -> None:
    points = [(p, rows[p][x_key], rows[p][y_key]) for p in POLICIES if p in rows]
    x_values = [p[1] for p in points]
    y_values = [p[2] for p in points]
    x_lo, x_hi = min(x_values), max(x_values)
    y_lo, y_hi = min(y_values), max(y_values)
    x_pad = max(1e-9, (x_hi - x_lo) * 0.14)
    y_pad = max(1e-9, (y_hi - y_lo) * 0.18)
    x_lo, x_hi = x_lo - x_pad, x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad
    plot_l, plot_r = x0 + 64, x0 + width - 26
    plot_t, plot_b = y0 + 42, y0 + height - 56

    lines.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 22:.1f}" text-anchor="middle" font-size="17" font-weight="700">{title}</text>')
    lines.append(f'<line x1="{plot_l:.1f}" y1="{plot_b:.1f}" x2="{plot_r:.1f}" y2="{plot_b:.1f}" stroke="#263241" stroke-width="1"/>')
    lines.append(f'<line x1="{plot_l:.1f}" y1="{plot_t:.1f}" x2="{plot_l:.1f}" y2="{plot_b:.1f}" stroke="#263241" stroke-width="1"/>')

    for tick in axis_ticks(x_lo, x_hi):
        x = scale(tick, x_lo, x_hi, plot_l, plot_r)
        lines.append(f'<line x1="{x:.1f}" y1="{plot_b:.1f}" x2="{x:.1f}" y2="{plot_b + 5:.1f}" stroke="#263241"/>')
        lines.append(f'<text x="{x:.1f}" y="{plot_b + 22:.1f}" text-anchor="middle" font-size="11" fill="#4b5563">{tick:.0f}</text>')
    for tick in axis_ticks(y_lo, y_hi):
        y = scale(tick, y_lo, y_hi, plot_b, plot_t)
        lines.append(f'<line x1="{plot_l - 5:.1f}" y1="{y:.1f}" x2="{plot_l:.1f}" y2="{y:.1f}" stroke="#263241"/>')
        lines.append(f'<text x="{plot_l - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{tick:.{y_digits}f}</text>')

    lines.append(f'<text x="{(plot_l + plot_r) / 2:.1f}" y="{y0 + height - 12:.1f}" text-anchor="middle" font-size="12" fill="#374151">{x_label}</text>')
    lines.append(f'<text x="{x0 + 14:.1f}" y="{(plot_t + plot_b) / 2:.1f}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90 {x0 + 14:.1f} {(plot_t + plot_b) / 2:.1f})">{y_label}</text>')

    label_offsets = {
        "full_rebalance": (8, -10),
        "rolling_horizon_rebalance": (8, 16),
        "eda_hourly_mo_no_reuse": (8, -12),
        "adr_eda_station_mo": (8, 16),
        "adr_eda_station": (8, 16),
    }
    for policy, x_value, y_value in points:
        x = scale(x_value, x_lo, x_hi, plot_l, plot_r)
        y = scale(y_value, y_lo, y_hi, plot_b, plot_t)
        color = COLORS[policy]
        radius = 7 if policy == "adr_eda_station_mo" else 5
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        dx, dy = label_offsets.get(policy, (8, -8))
        weight = "700" if policy == "adr_eda_station_mo" else "500"
        lines.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="11" font-weight="{weight}" fill="{color}">{LABELS[policy]}</text>')


def build_svg(
    path: Path,
    datasets: dict[str, dict[str, dict[str, float]]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    y_digits: int,
) -> None:
    dataset_items = list(datasets.items())
    panel_w = 600
    width = 40 + len(dataset_items) * 620
    height = 520
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{width / 2}" y="54" text-anchor="middle" font-size="13" fill="#4b5563">RealBike h336, EDAspy/SPEDA, distancias OSRM, predicción histórica en línea</text>',
    ]
    for idx, (name, rows) in enumerate(dataset_items):
        draw_panel(
            lines,
            30 + 620 * idx,
            80,
            panel_w,
            390,
            name,
            rows,
            x_key,
            y_key,
            x_label,
            y_label,
            y_digits,
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_csv(path: Path, datasets: dict[str, dict[str, dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "policy",
        "utility",
        "service",
        "rejected",
        "km_osrm",
        "bikes",
        "reopt",
        "reuse",
        "decision_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rows in datasets.items():
            for policy in POLICIES:
                if policy not in rows:
                    continue
                row = rows[policy]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "policy": LABELS[policy],
                        "utility": f"{row['utility_score_mean']:.2f}",
                        "service": f"{row['service_rate_mean']:.4f}",
                        "rejected": f"{row['rejected_mean']:.2f}",
                        "km_osrm": f"{row['relocation_km_mean']:.2f}",
                        "bikes": f"{row['relocated_bikes_mean']:.2f}",
                        "reopt": f"{row['reoptimization_epochs_mean']:.2f}",
                        "reuse": f"{row['reuse_epochs_mean']:.2f}",
                        "decision_sec": f"{row['decision_time_sec_mean']:.2f}",
                    }
                )


def write_latex_table(path: Path, datasets: dict[str, dict[str, dict[str, float]]]) -> None:
    lines = [
        "% Auto-generated RealBike h336 table; labels adapted for thesis readability.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Resultados RealBike h336 con EDAspy/SPEDA, OSRM y predicción histórica en línea.}",
        "\\label{tab:realbike_h336_multicity}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Sistema & Política & Utilidad & Servicio & Rech. & Km & Bicis & Reopt. & Reutil. & Decisión (s) \\\\",
        "\\midrule",
    ]
    for dataset, rows in datasets.items():
        for policy in POLICIES:
            if policy not in rows:
                continue
            row = rows[policy]
            lines.append(
                f"{dataset} & {LABELS[policy]} & "
                f"{row['utility_score_mean']:.2f} & {row['service_rate_mean']:.4f} & "
                f"{row['rejected_mean']:.1f} & {row['relocation_km_mean']:.1f} & "
                f"{row['relocated_bikes_mean']:.1f} & {row['reoptimization_epochs_mean']:.1f} & "
                f"{row['reuse_epochs_mean']:.1f} & {row['decision_time_sec_mean']:.2f} \\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pairwise_csv(
    path: Path,
    pairwise: dict[str, list[dict[str, str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "comparison",
        "delta_utility",
        "delta_service",
        "delta_km_osrm",
        "delta_reopt",
        "utility_wins",
        "service_nonworse",
        "km_wins",
        "reopt_wins",
        "paired_seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rows in pairwise.items():
            for rhs in ["full_rebalance", "rolling_horizon_rebalance", "eda_hourly_mo_no_reuse"]:
                row = find_pair(rows, "adr_eda_station_mo", rhs)
                if row is None:
                    continue
                writer.writerow(
                    {
                        "dataset": dataset,
                        "comparison": f"ADR-EDA frente a {PAIR_LABELS[rhs]}",
                        "delta_utility": f"{float(row['utility_delta_mean']):.2f}",
                        "delta_service": f"{float(row['service_delta_mean']):.4f}",
                        "delta_km_osrm": f"{float(row['relocation_km_delta_mean']):.2f}",
                        "delta_reopt": f"{float(row['reoptimization_delta_mean']):.2f}",
                        "utility_wins": row["utility_wins"],
                        "service_nonworse": row["service_nonworse"],
                        "km_wins": row["relocation_km_wins"],
                        "reopt_wins": row["reoptimization_wins"],
                        "paired_seeds": row["paired_seeds"],
                    }
                )


def write_pairwise_latex(
    path: Path,
    pairwise: dict[str, list[dict[str, str]]],
) -> None:
    lines = [
        "% Auto-generated RealBike h336 pairwise table; labels adapted for thesis readability.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Comparación pareada de ADR-EDA frente a referencias RealBike h336.}",
        "\\label{tab:realbike_h336_multicity_pairwise}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Sistema & Comparación & $\\Delta$ utilidad & $\\Delta$ servicio & $\\Delta$ km OSRM & $\\Delta$ reopt. \\\\",
        "\\midrule",
    ]
    for dataset, rows in pairwise.items():
        for rhs in ["full_rebalance", "rolling_horizon_rebalance", "eda_hourly_mo_no_reuse"]:
            row = find_pair(rows, "adr_eda_station_mo", rhs)
            if row is None:
                continue
            lines.append(
                f"{dataset} & ADR-EDA frente a {PAIR_LABELS[rhs]} & "
                f"{float(row['utility_delta_mean']):.2f} & "
                f"{float(row['service_delta_mean']):.4f} & "
                f"{float(row['relocation_km_delta_mean']):.2f} & "
                f"{float(row['reoptimization_delta_mean']):.2f} \\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_narrative(
    path: Path,
    datasets: dict[str, dict[str, dict[str, float]]],
    pairwise: dict[str, list[dict[str, str]]],
) -> None:
    system_names = ", ".join(datasets.keys())
    lines = [
        "# Paquete de tesis RealBike h336",
        "",
        f"Este paquete conserva la evidencia RealBike usada en la memoria: {system_names} con demanda real a nivel de estación, matrices OSRM, predicción `historical_hour` en línea, calentamiento de 72 horas, horizonte h336 y EDAspy/SPEDA para las políticas EDA.",
        "",
        "## Lectura central",
        "",
        "ADR-EDA debe leerse como un controlador adaptativo consciente del coste, que decide cuándo actualizar y cuándo reutilizar el objetivo aprendido. Las políticas que recalculan cada hora pueden ganar servicio moviendo más distancia; la aportación de ADR-EDA es mantener un servicio competitivo reduciendo muchas actualizaciones completas.",
        "",
        "## Lectura por sistema",
        "",
    ]
    for dataset, rows in datasets.items():
        adr = rows["adr_eda_station_mo"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Utilidad ADR-EDA: {adr['utility_score_mean']:.2f}.",
                f"- Servicio ADR-EDA: {adr['service_rate_mean']:.4f}.",
                f"- Kilómetros OSRM ADR-EDA: {adr['relocation_km_mean']:.2f}.",
                f"- Reoptimizaciones/reutilizaciones ADR-EDA: {adr['reoptimization_epochs_mean']:.1f}/{adr['reuse_epochs_mean']:.1f}.",
                "",
            ]
        )
        for rhs in ["full_rebalance", "rolling_horizon_rebalance", "eda_hourly_mo_no_reuse"]:
            row = find_pair(pairwise[dataset], "adr_eda_station_mo", rhs)
            if row is None:
                continue
            lines.append(
                f"- Frente a {PAIR_LABELS[rhs]}: delta de utilidad {float(row['utility_delta_mean']):.2f}, "
                f"delta de servicio {float(row['service_delta_mean']):.4f}, "
                f"delta de km {float(row['relocation_km_delta_mean']):.2f}, "
                f"delta de reoptimizaciones {float(row['reoptimization_delta_mean']):.2f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Encaje en la memoria",
            "",
            f"- {system_names} constituyen la evidencia multiciudad principal de RealBike en la memoria.",
            "- La formulación más sólida es: ADR-EDA reduce de forma sustancial las reoptimizaciones completas y ofrece una frontera reproducible entre servicio, coste y esfuerzo computacional sobre demanda real de bicicletas compartidas a nivel de estación.",
            "",
            "## Archivos",
            "",
            "- `realbike_h336_summary_table.csv` / `.tex`: indicadores absolutos por política.",
            "- `realbike_h336_pairwise_deltas.csv` / `.tex`: tabla central de deltas pareados.",
            "- `realbike_h336_service_vs_km.svg` / `.png`: figura de servicio y coste.",
            "- `realbike_h336_reopt_vs_decision_time.svg` / `.png`: figura de esfuerzo computacional.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divvy-summary", required=True, type=Path)
    parser.add_argument("--divvy-pairwise", required=True, type=Path)
    parser.add_argument("--citibike-summary", required=True, type=Path)
    parser.add_argument("--citibike-pairwise", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    datasets = {
        "Divvy": read_summary(args.divvy_summary),
        "Citi Bike": read_summary(args.citibike_summary),
    }
    pairwise = {
        "Divvy": read_pairwise(args.divvy_pairwise),
        "Citi Bike": read_pairwise(args.citibike_pairwise),
    }

    build_svg(
        args.out_dir / "realbike_h336_service_vs_km.svg",
        datasets,
        x_key="relocation_km_mean",
        y_key="service_rate_mean",
        title="RealBike: compromiso servicio-coste",
        x_label="Distancia de reequilibrado (km OSRM)",
        y_label="Tasa de servicio",
        y_digits=4,
    )
    build_svg(
        args.out_dir / "realbike_h336_reopt_vs_decision_time.svg",
        datasets,
        x_key="reoptimization_epochs_mean",
        y_key="decision_time_sec_mean",
        title="RealBike: reoptimizaciones y tiempo de decisión",
        x_label="Reoptimizaciones completas",
        y_label="Tiempo de decisión (s)",
        y_digits=0,
    )
    write_table_csv(args.out_dir / "realbike_h336_summary_table.csv", datasets)
    write_latex_table(args.out_dir / "realbike_h336_summary_table.tex", datasets)
    write_pairwise_csv(args.out_dir / "realbike_h336_pairwise_deltas.csv", pairwise)
    write_pairwise_latex(args.out_dir / "realbike_h336_pairwise_deltas.tex", pairwise)
    write_narrative(args.out_dir / "REALBIKE_H336_THESIS_PACKAGE.md", datasets, pairwise)
    print(f"Wrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
