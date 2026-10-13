#!/usr/bin/env python3
"""Build the final RealBike thesis/paper package from existing experiment CSVs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REALBIKE = ROOT / "docs" / "realbike_dmoo"
OUT = REALBIKE / "final_realbike_package_2026_05_25"
PAPER_ALIGNED = REALBIKE / "paper_aligned_citibike_osrm_static_s10_2026_05_25"
EVENT_MULTI = REALBIKE / "event_replay_multicity_osrm_s10_h168_2026_05_25"


POLICY_ROWS = [
    {
        "policy": "static_twice_daily_rebalance",
        "short_name": "static 2/day",
        "reoptimizes": "03:00 y 15:00",
        "reuses": "no",
        "uses_eda": "no",
        "role": "baseline estatico inspirado en literatura",
    },
    {
        "policy": "full_rebalance",
        "short_name": "full hourly",
        "reoptimizes": "cada hora",
        "reuses": "no",
        "uses_eda": "no",
        "role": "baseline determinista agresivo",
    },
    {
        "policy": "rolling_horizon_rebalance",
        "short_name": "rolling horizon",
        "reoptimizes": "cada hora",
        "reuses": "no",
        "uses_eda": "no",
        "role": "baseline operativo tipo MPC",
    },
    {
        "policy": "eda_hourly_no_reuse",
        "short_name": "EDA hourly",
        "reoptimizes": "cada hora",
        "reuses": "no",
        "uses_eda": "si",
        "role": "cota agresiva sin ADR",
    },
    {
        "policy": "adr_eda_station",
        "short_name": "ADR-EDA",
        "reoptimizes": "adaptativo",
        "reuses": "si",
        "uses_eda": "si",
        "role": "politica propuesta",
    },
]


COLORS = {
    "no_relocation": "#7f8c8d",
    "static_twice_daily_rebalance": "#7b5ea7",
    "full_rebalance": "#d55e00",
    "rolling_horizon_rebalance": "#0072b2",
    "selective_rebalance": "#009e73",
    "adr_safe_reuse_proxy": "#cc79a7",
    "eda_hourly_no_reuse": "#56b4e9",
    "adr_eda_station": "#111111",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def latex_escape(value: object) -> str:
    text = str(value)
    for src, dst in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[object]],
    size: str = r"\scriptsize",
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        size,
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(headers) + r"}",
        r"\hline",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def to_policy_label(policy: str) -> str:
    return {
        "no_relocation": "sin recolocacion",
        "static_twice_daily_rebalance": "static 2/day",
        "full_rebalance": "full horario",
        "rolling_horizon_rebalance": "rolling horizon",
        "selective_rebalance": "selective",
        "adr_safe_reuse_proxy": "ADR proxy",
        "eda_hourly_no_reuse": "EDA horaria",
        "adr_eda_station": "ADR-EDA",
    }.get(policy, policy)


def svg_scatter(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[dict[str, object]],
    width: int = 920,
    height: int = 560,
) -> None:
    margin_l = 92
    margin_r = 38
    margin_t = 64
    margin_b = 78
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    x_min = 0.0
    x_max = max(xs) * 1.08 if xs else 1.0
    y_min = min(ys) - max(0.001, (max(ys) - min(ys)) * 0.2)
    y_max = min(1.001, max(ys) + max(0.001, (max(ys) - min(ys)) * 0.2))
    if y_max <= y_min:
        y_min -= 0.001
        y_max += 0.001

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#333" stroke-width="1.4"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#333" stroke-width="1.4"/>',
    ]
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        px = sx(x)
        lines.append(f'<line x1="{px:.1f}" y1="{margin_t}" x2="{px:.1f}" y2="{height-margin_b}" stroke="#e6e6e6"/>')
        lines.append(f'<text x="{px:.1f}" y="{height-margin_b+22}" text-anchor="middle" font-family="Arial" font-size="12">{x:.0f}</text>')
    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        py = sy(y)
        lines.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{width-margin_r}" y2="{py:.1f}" stroke="#e6e6e6"/>')
        lines.append(f'<text x="{margin_l-10}" y="{py+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.3f}</text>')
    lines.append(f'<text x="{width/2:.1f}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(x_label)}</text>')
    lines.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 24 {height/2:.1f})">{html.escape(y_label)}</text>'
    )

    for p in points:
        policy = str(p["policy"])
        px = sx(float(p["x"]))
        py = sy(float(p["y"]))
        color = COLORS.get(policy, "#333333")
        radius = 8 if policy == "adr_eda_station" else 6
        stroke = "#000000" if policy == "adr_eda_station" else "#ffffff"
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        dx = 10
        dy = -10 if policy in {"adr_eda_station", "eda_hourly_no_reuse", "rolling_horizon_rebalance"} else 16
        label = str(p.get("label", to_policy_label(policy)))
        lines.append(
            f'<text x="{px+dx:.1f}" y="{py+dy:.1f}" font-family="Arial" font-size="13" font-weight="{700 if policy == "adr_eda_station" else 400}">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_policy_tables() -> None:
    write_csv(OUT / "tables" / "policy_table.csv", POLICY_ROWS)
    write_latex_table(
        OUT / "tables" / "policy_table.tex",
        caption="Politicas comparadas en RealBike.",
        label="tab:realbike-politicas",
        headers=["Politica", "Reoptimiza", "Reutiliza", "Usa EDA", "Rol"],
        rows=[
            [
                row["short_name"],
                row["reoptimizes"],
                row["reuses"],
                row["uses_eda"],
                row["role"],
            ]
            for row in POLICY_ROWS
        ],
    )


def build_citibike_tables() -> list[dict[str, object]]:
    summary = read_csv(PAPER_ALIGNED / "summary.csv")
    rows = []
    for row in summary:
        policy = row["policy"]
        rows.append(
            {
                "policy": policy,
                "label": to_policy_label(policy),
                "utility": round(float(row["utility_score_mean"]), 3),
                "service_rate": round(float(row["service_rate_mean"]), 6),
                "relocation_km": round(float(row["relocation_km_mean"]), 3),
                "reoptimizations": round(float(row["reoptimization_epochs_mean"]), 3),
                "reuse_epochs": round(float(row["reuse_epochs_mean"]), 3),
                "rejected": round(float(row["rejected_mean"]), 3),
            }
        )
    write_csv(OUT / "tables" / "citibike_osrm_main_results.csv", rows)
    write_latex_table(
        OUT / "tables" / "citibike_osrm_main_results.tex",
        caption="Resultados principales CitiBike con matriz OSRM, 336 horas y 10 semillas.",
        label="tab:realbike-citibike-osrm",
        headers=["Politica", "Utilidad", "Servicio", "Km", "Reopt.", "Reuse", "Rechazos"],
        rows=[
            [
                row["label"],
                fmt(float(row["utility"]), 1),
                fmt(float(row["service_rate"]), 4),
                fmt(float(row["relocation_km"]), 1),
                fmt(float(row["reoptimizations"]), 1),
                fmt(float(row["reuse_epochs"]), 1),
                fmt(float(row["rejected"]), 1),
            ]
            for row in rows
        ],
    )

    pairwise = read_csv(PAPER_ALIGNED / "pairwise.csv")
    keep = {"static_twice_daily_rebalance", "full_rebalance", "rolling_horizon_rebalance", "eda_hourly_no_reuse"}
    pair_rows = []
    for row in pairwise:
        if row["lhs"] == "adr_eda_station" and row["rhs"] in keep:
            pair_rows.append(
                {
                    "comparison": "ADR-EDA vs " + to_policy_label(row["rhs"]),
                    "utility_delta": round(float(row["utility_delta_mean"]), 3),
                    "utility_wins": int(row["utility_wins"]),
                    "service_delta": round(float(row["service_delta_mean"]), 6),
                    "km_delta": round(float(row["relocation_km_delta_mean"]), 3),
                    "km_saved": round(-float(row["relocation_km_delta_mean"]), 3),
                    "reoptimization_delta": round(float(row["reoptimization_delta_mean"]), 3),
                    "reopt_saved": round(-float(row["reoptimization_delta_mean"]), 3),
                }
            )
    write_csv(OUT / "tables" / "citibike_osrm_adr_pairwise.csv", pair_rows)
    write_latex_table(
        OUT / "tables" / "citibike_osrm_adr_pairwise.tex",
        caption="Comparacion pareada de ADR-EDA frente a baselines CitiBike OSRM.",
        label="tab:realbike-citibike-pairwise",
        headers=["Comparacion", "Delta utilidad", "Delta servicio", "Delta km", "Delta reopt."],
        rows=[
            [
                row["comparison"],
                fmt(float(row["utility_delta"]), 1),
                fmt(float(row["service_delta"]), 4),
                fmt(float(row["km_delta"]), 1),
                fmt(float(row["reoptimization_delta"]), 1),
            ]
            for row in pair_rows
        ],
    )
    svg_scatter(
        OUT / "figures" / "citibike_osrm_service_cost_frontier.svg",
        title="CitiBike OSRM: frontera servicio-coste",
        x_label="Kilometros de recolocacion",
        y_label="Tasa de servicio",
        points=[
            {"policy": row["policy"], "x": row["relocation_km"], "y": row["service_rate"]}
            for row in rows
        ],
    )
    return rows


def event_metrics(path: Path) -> dict[str, dict[str, float]]:
    by_policy: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        by_policy.setdefault(row["policy"], {})[row["metric"]] = float(row["mean"])
    return by_policy


def build_event_tables() -> list[dict[str, object]]:
    out = []
    for dataset in ["citibike", "divvy"]:
        for gate in ["gate1_4_aggressive", "gate2_8_reuse"]:
            metrics = event_metrics(EVENT_MULTI / f"{dataset}_{gate}_event" / "event_summary.csv")
            adr = metrics["adr_eda_station"]
            rolling = metrics["rolling_horizon_rebalance"]
            static = metrics["static_twice_daily_rebalance"]
            out.append(
                {
                    "dataset": dataset,
                    "gate": "1/4 agresivo" if gate.startswith("gate1") else "2/8 reutilizador",
                    "adr_service": round(adr["service_rate"], 6),
                    "adr_utility": round(adr["utility_score"], 3),
                    "adr_km": round(adr["relocation_km"], 3),
                    "adr_reoptimizations": round(adr["reoptimization_epochs"], 3),
                    "adr_reuse": round(adr["reuse_epochs"], 3),
                    "utility_delta_vs_rolling": round(adr["utility_score"] - rolling["utility_score"], 3),
                    "km_saved_vs_rolling": round(rolling["relocation_km"] - adr["relocation_km"], 3),
                    "service_delta_vs_static": round(adr["service_rate"] - static["service_rate"], 6),
                    "utility_delta_vs_static": round(adr["utility_score"] - static["utility_score"], 3),
                }
            )
    write_csv(OUT / "tables" / "event_replay_multicity_results.csv", out)
    write_latex_table(
        OUT / "tables" / "event_replay_multicity_results.tex",
        caption="Resultados evento-a-evento con CitiBike y Divvy, 168 horas, 10 semillas y OSRM.",
        label="tab:realbike-event-replay",
        headers=["Dataset", "Gate", "Servicio ADR", "Utilidad ADR", "Km ADR", "Reopt. ADR", "Km ahorrados vs rolling"],
        rows=[
            [
                row["dataset"],
                row["gate"],
                fmt(float(row["adr_service"]), 4),
                fmt(float(row["adr_utility"]), 1),
                fmt(float(row["adr_km"]), 1),
                fmt(float(row["adr_reoptimizations"]), 1),
                fmt(float(row["km_saved_vs_rolling"]), 1),
            ]
            for row in out
        ],
    )

    gap_rows = []
    for dataset in ["citibike", "divvy"]:
        for gate in ["gate1_4_aggressive", "gate2_8_reuse"]:
            for row in read_csv(EVENT_MULTI / f"{dataset}_{gate}_event" / "aggregation_gap.csv"):
                if row["policy"] != "adr_eda_station":
                    continue
                if row["metric"] not in {"utility_score", "service_rate", "relocation_km", "reoptimization_epochs", "reuse_epochs", "rejected"}:
                    continue
                gap_rows.append(
                    {
                        "dataset": dataset,
                        "gate": "1/4 agresivo" if gate.startswith("gate1") else "2/8 reutilizador",
                        "metric": row["metric"],
                        "hourly_mean": round(float(row["hourly_mean"]), 6),
                        "event_mean": round(float(row["event_mean"]), 6),
                        "event_minus_hourly": round(float(row["event_minus_hourly"]), 6),
                    }
                )
    write_csv(OUT / "tables" / "event_replay_adr_aggregation_gap.csv", gap_rows)

    points = []
    for dataset in ["citibike", "divvy"]:
        for gate in ["gate1_4_aggressive", "gate2_8_reuse"]:
            metrics = event_metrics(EVENT_MULTI / f"{dataset}_{gate}_event" / "event_summary.csv")
            for policy in ["static_twice_daily_rebalance", "rolling_horizon_rebalance", "adr_eda_station"]:
                suffix = " 1/4" if gate.startswith("gate1") else " 2/8"
                points.append(
                    {
                        "policy": policy,
                        "label": f"{dataset} {to_policy_label(policy)}{suffix}",
                        "x": metrics[policy]["relocation_km"],
                        "y": metrics[policy]["service_rate"],
                    }
                )
    svg_scatter(
        OUT / "figures" / "event_replay_multicity_service_cost_frontier.svg",
        title="Replay cronologico: servicio-coste",
        x_label="Kilometros de recolocacion",
        y_label="Tasa de servicio",
        points=points,
    )
    return out


def build_narrative(citibike_rows: list[dict[str, object]], event_rows: list[dict[str, object]]) -> None:
    adr = next(row for row in citibike_rows if row["policy"] == "adr_eda_station")
    static = next(row for row in citibike_rows if row["policy"] == "static_twice_daily_rebalance")
    rolling = next(row for row in citibike_rows if row["policy"] == "rolling_horizon_rebalance")
    eda = next(row for row in citibike_rows if row["policy"] == "eda_hourly_no_reuse")
    citi_gate14 = next(row for row in event_rows if row["dataset"] == "citibike" and row["gate"].startswith("1/4"))
    divvy_gate14 = next(row for row in event_rows if row["dataset"] == "divvy" and row["gate"].startswith("1/4"))
    citi_gate28 = next(row for row in event_rows if row["dataset"] == "citibike" and row["gate"].startswith("2/8"))
    divvy_gate28 = next(row for row in event_rows if row["dataset"] == "divvy" and row["gate"].startswith("2/8"))

    md = f"""# Paquete final RealBike

Fecha: 2026-05-25

Este paquete cierra RealBike como experimento central de tesis/paper. La idea es dejar de abrir lineas nuevas y presentar una historia compacta:

> ADR-EDA como politica coste-consciente para reequilibrado dinamico station-based, evaluada en CitiBike/Divvy con distancias OSRM y comparada contra baselines alineados con literatura.

## Archivos generados

- `tables/policy_table.csv` y `tables/policy_table.tex`
- `tables/citibike_osrm_main_results.csv` y `.tex`
- `tables/citibike_osrm_adr_pairwise.csv` y `.tex`
- `tables/event_replay_multicity_results.csv` y `.tex`
- `tables/event_replay_adr_aggregation_gap.csv`
- `figures/citibike_osrm_service_cost_frontier.svg`
- `figures/event_replay_multicity_service_cost_frontier.svg`
- `realbike_tfm_section_snippet.tex`

## Mensaje central

El reequilibrado dinamico mejora el servicio, pero reoptimizar cada hora puede ser caro. ADR-EDA busca una posicion intermedia: aprender una distribucion de inventarios objetivo y decidir cuando actualizarla o reutilizarla.

En la campana CitiBike OSRM de 336 horas y 10 semillas, ADR-EDA alcanza servicio {fmt(float(adr['service_rate']), 4)}, frente a {fmt(float(static['service_rate']), 4)} del baseline estatico y {fmt(float(rolling['service_rate']), 4)} de rolling horizon. Frente al estatico mejora la utilidad en {fmt(float(adr['utility']) - float(static['utility']), 1)} puntos. Frente a rolling horizon queda por debajo en utilidad, pero ahorra {fmt(float(rolling['relocation_km']) - float(adr['relocation_km']), 1)} km y {fmt(float(rolling['reoptimizations']) - float(adr['reoptimizations']), 1)} reoptimizaciones.

Frente a EDA horaria, ADR-EDA pierde algo de calidad, pero ahorra {fmt(float(eda['reoptimizations']) - float(adr['reoptimizations']), 1)} reoptimizaciones y {fmt(float(eda['relocation_km']) - float(adr['relocation_km']), 1)} km. Esta es la comparacion que justifica el gate adaptativo.

## Validacion cronologica evento-a-evento

La campana multicity evento-a-evento refuerza que RealBike no es solo un benchmark agregado. En CitiBike con gate 1/4, ADR-EDA obtiene servicio {fmt(float(citi_gate14['adr_service']), 4)}, queda practicamente empatado con rolling horizon en utilidad y ahorra {fmt(float(citi_gate14['km_saved_vs_rolling']), 1)} km. En Divvy con gate 1/4, ADR-EDA obtiene servicio {fmt(float(divvy_gate14['adr_service']), 4)}, supera a rolling horizon en utilidad por {fmt(float(divvy_gate14['utility_delta_vs_rolling']), 1)} y ahorra {fmt(float(divvy_gate14['km_saved_vs_rolling']), 1)} km.

El gate 2/8 es la version mas orientada a reutilizacion: en CitiBike baja a {fmt(float(citi_gate28['adr_reoptimizations']), 1)} reoptimizaciones y {fmt(float(citi_gate28['adr_reuse']), 1)} reutilizaciones; en Divvy baja a {fmt(float(divvy_gate28['adr_reoptimizations']), 1)} reoptimizaciones y {fmt(float(divvy_gate28['adr_reuse']), 1)} reutilizaciones. La perdida existe, pero no destruye el servicio.

## Como conectarlo con CAEPIA

La frase recomendada es:

> En CAEPIA se evaluo el principio ADR-EDA en benchmarks dinamicos generales. En RealBike se instancia ese principio en un dominio station-level: la solucion pasa a ser un vector de inventario objetivo por estacion y el gate adaptativo se alimenta de presion de demanda, drift del target y servicio reciente. Esto muestra que ADR-EDA no es una heuristica rigida, sino una arquitectura adaptable a distintos dominios dinamicos.

## Que afirmar

- RealBike usa datos reales de CitiBike/Divvy.
- Las distancias OSRM mejoran el realismo del coste de recolocacion.
- ADR-EDA mejora claramente al baseline estatico.
- ADR-EDA se acerca a rolling/EDA horaria con menos km y/o menos reoptimizaciones.
- El gate permite una frontera interpretable entre rendimiento y reutilizacion.
- El replay evento-a-evento muestra que el resultado no depende solo de una simulacion horaria agregada.

## Que no afirmar

- No afirmar que se reproduce exactamente la operacion industrial de CitiBike o Divvy.
- No afirmar que ADR-EDA domina universalmente a todos los baselines.
- No afirmar que OSRM equivale a trafico real en vivo.
- No afirmar que ya se resuelve el problema completo de rutas de camiones.

## Decision recomendada

Para tesis, usar `gate 1/4` como configuracion de mayor rendimiento y `gate 2/8` como analisis de sensibilidad que demuestra la palanca de reutilizacion. No abrir rutas estocasticas ni free-floating hasta que esta historia este escrita y visualmente cerrada.
"""
    (OUT / "REALBIKE_FINAL_NARRATIVE_2026_05_25.md").write_text(md, encoding="utf-8")

    tex = r"""\section{Validacion RealBike con datos reales y OSRM}
\label{sec:realbike-final}

RealBike se plantea como una validacion externa de ADR-EDA sobre datos reales de bike-sharing station-based. A diferencia de los benchmarks sinteticos, aqui las estaciones, coordenadas y viajes proceden de CitiBike y Divvy. La decision optimizada por ADR-EDA es un vector de inventario objetivo por estacion; la rutina de recolocacion traduce ese vector en movimientos de bicicletas, y la demanda observada actualiza el estado del sistema.

La motivacion es operacional: reoptimizar cada hora puede mejorar el servicio, pero tambien incrementa los kilometros de recolocacion y el coste computacional. ADR-EDA introduce una puerta adaptativa que decide cuando actualizar la distribucion aprendida y cuando reutilizar la solucion heredada.

\input{../docs/realbike_dmoo/final_realbike_package_2026_05_25/tables/policy_table.tex}

\input{../docs/realbike_dmoo/final_realbike_package_2026_05_25/tables/citibike_osrm_main_results.tex}

En la campana CitiBike con matriz OSRM, ADR-EDA supera claramente al baseline estatico dos veces al dia y se aproxima a las politicas dinamicas agresivas. Frente a rolling horizon reduce kilometros y reoptimizaciones, aunque con una ligera perdida de servicio. Frente a EDA horaria sin reutilizacion, ADR-EDA mantiene un nivel de servicio alto con menos actualizaciones, lo que justifica la puerta adaptativa.

\input{../docs/realbike_dmoo/final_realbike_package_2026_05_25/tables/citibike_osrm_adr_pairwise.tex}

Adicionalmente, siguiendo la literatura reciente sobre reequilibrado dinamico, se ejecuta una validacion cronologica evento-a-evento. Esta evaluacion evita depender exclusivamente de demanda agregada por hora: cada alquiler y devolucion se aplica en su instante observado, mientras que la politica sigue decidiendo en fronteras horarias.

\input{../docs/realbike_dmoo/final_realbike_package_2026_05_25/tables/event_replay_multicity_results.tex}

Los resultados evento-a-evento muestran que ADR-EDA no colapsa al pasar a una simulacion mas fina. En CitiBike, con gate agresivo, iguala el servicio de rolling horizon y reduce kilometros. En Divvy, el mismo gate mejora la utilidad frente a rolling horizon y tambien reduce kilometros. El gate mas reutilizador reduce aproximadamente a la mitad las reoptimizaciones, a costa de una perdida moderada, lo que ofrece una palanca interpretable para ajustar la politica al coste operativo.

La conclusion defendible es que ADR-EDA funciona como politica coste-consciente para reequilibrado dinamico station-based: aprende distribuciones de inventario objetivo y decide cuando actualizarlas o reutilizarlas. Este bloque no pretende reproducir todos los detalles industriales de una empresa de bike-sharing, pero si aporta una validacion realista, reproducible y alineada con literatura mediante datos reales, distancias OSRM, baselines operativos y replay cronologico de eventos.
"""
    (OUT / "realbike_tfm_section_snippet.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    build_policy_tables()
    citibike_rows = build_citibike_tables()
    event_rows = build_event_tables()
    build_narrative(citibike_rows, event_rows)
    manifest = {
        "created_from": {
            "paper_aligned": str(PAPER_ALIGNED),
            "event_multicity": str(EVENT_MULTI),
        },
        "outputs": sorted(str(path.relative_to(OUT)) for path in OUT.rglob("*") if path.is_file()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
