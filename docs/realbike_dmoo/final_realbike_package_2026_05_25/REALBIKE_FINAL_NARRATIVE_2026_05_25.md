# Paquete final RealBike

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

En la campana CitiBike OSRM de 336 horas y 10 semillas, ADR-EDA alcanza servicio 0.9902, frente a 0.9743 del baseline estatico y 0.9953 de rolling horizon. Frente al estatico mejora la utilidad en 467.3 puntos. Frente a rolling horizon queda por debajo en utilidad, pero ahorra 1785.5 km y 137.4 reoptimizaciones.

Frente a EDA horaria, ADR-EDA pierde algo de calidad, pero ahorra 137.4 reoptimizaciones y 582.0 km. Esta es la comparacion que justifica el gate adaptativo.

## Validacion cronologica evento-a-evento

La campana multicity evento-a-evento refuerza que RealBike no es solo un benchmark agregado. En CitiBike con gate 1/4, ADR-EDA obtiene servicio 1.0000, supera ligeramente a rolling horizon en utilidad y ahorra 440.8 km. En Divvy con gate 1/4, ADR-EDA obtiene servicio 0.9804, supera a rolling horizon en utilidad por 52.9 y ahorra 920.7 km.

El gate 2/8 es la version mas orientada a reutilizacion: en CitiBike baja a 84.0 reoptimizaciones y 84.0 reutilizaciones; en Divvy baja a 89.9 reoptimizaciones y 78.1 reutilizaciones. La perdida existe, pero no destruye el servicio.

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
