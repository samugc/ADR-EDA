# free_floating_robust

Implementacion activa del modelo one-way free-floating EV usado en la memoria
del TFM.

## Modulos principales

- `types.py`: entidades de dominio, vehiculos, tareas, solicitudes y metricas.
- `dynamics.py`: transiciones de estado, viajes, carga y relocalizacion.
- `scenario.py`: escenarios sinteticos y perfiles de demanda.
- `travel_metrics.py`: lectura de matrices de movilidad y congestion.
- `ops_cost_profiles.py`: perfiles de coste operativo.
- `planner.py`: controlador rolling horizon.
- `evaluator.py`: evaluador multiobjetivo.
- `edaspy_bridge.py`: puente EDAspy clasico.
- `edaspy_dynamic_bridge.py`: puente DynamicEDA/ADR-EDA.
- `edaspy_runtime.py`: resolucion local/Docker de EDAspy.
- `live_requests_demo.py`: motor interno de simulacion online usado por
  `runner.py`.
- `runner.py`: punto principal de ejecucion.

## Ejecucion rapida

Desde la raiz del repositorio:

```bash
PYTHONPATH=src python3 -m free_floating_robust.runner --help
```

## Modo aplicado principal

El modo `online-eda` es el mas cercano a la validacion aplicada: congela el
estado cada intervalo de decision, evalua si conviene reoptimizar y conserva lo
que ya esta ejecutandose.

Ejemplo corto:

```bash
PYTHONPATH=src python3 -m free_floating_robust.runner \
  --mode online-eda \
  --ops-cost-profile serio \
  --online-reopt-profile full \
  --sim-duration-min 60 \
  --horizon-hours 1 \
  --fleet-size 220 \
  --charger-count 22 \
  --ops-resources 26 \
  --size-gen 24 \
  --max-iter 2 \
  --dead-iter 2 \
  --output-json tests/model/results/runner_online_eda_full_serio.json
```

## Validacion reproducible

El runner aplicado que aparece en los anexos es:

```bash
PYTHONPATH=src python3 tests/model/run_online_full_vs_adr_solid_validation.py --help
```

La matriz OSRM para esa validacion se puede reconstruir con:

```bash
PYTHONPATH=src python3 tests/model/build_osrm_zone_matrix.py --help
```

## EDAspy

Los modos EDA buscan EDAspy en:

- `EDAspy/`
- `/app/EDAspy`

El checkout local de EDAspy no se versiona en este repositorio; debe estar
disponible en una de esas rutas o dentro del contenedor Docker usado para las
campanas originales.

## Baseline Madrid serio

Parametros operativos fijados por defecto en la lectura de tesis:

- `fleet_size = 220`
- `charger_count = 22`
- `ops_resources = 26`
- `reservation_share = 0.28`
- `reservation_booking_intensity_mult = 1.0`
- `immediate_request_intensity_mult = 1.0`
- `reservation_mean_lead_min = 35`
- `reservation_cancel_base_prob = 0.06`

Datos de demanda base:

- `src/free_floating_robust/data/madrid_zone_base_demand_serious_baseline.json`
- `src/free_floating_robust/data/madrid_zone_base_demand_core_weighted_v1.json`
- `src/free_floating_robust/data/madrid_zone_base_demand_legacy_neutral_baseline.json`

La salida economica se interpreta como margen operativo episodico, no como una
cuenta de resultados empresarial completa.
