# Carsharing thesis package

Paquete generado a partir de salidas consolidadas de las campañas. La evidencia
central del TFM corresponde a la campaña controlada de carsharing
free-floating; RealCar-Munich se conserva como extensión de transferencia con
demanda real observada.

## Artifacts

- `carsharing_synthetic_core_table.tex`: official synthetic 8 h table.
- `carsharing_synthetic_pairwise_deltas.tex`: paired deltas against full static and fast event.
- `carsharing_synthetic_pareto_snapshot.tex`: selected ADR-EDA objective snapshot.
- `realcar_munich_summary_table.tex`: real Munich OD demand extension.
- `realcar_munich_objective_table.tex`: Munich selected objective summary.

## Lectura

- El bloque sintético de carsharing es la evidencia aplicada central: ADR-EDA mejora margen, kilometraje improductivo y reoptimizaciones frente a la referencia estática, con una pequeña pérdida de servicio.
- La comparación con fast_event aporta evidencia de mecanismo: ADR-EDA mejora el compromiso medio frente a una referencia selectiva, aunque no todas las semillas mejoran simultáneamente.
- Mayor mejora de margen comparable en RealCar-Munich dentro de este paquete:
  10 zonas h12 s3, con delta de margen de 1117.24 €.
- Mayor mejora de servicio en RealCar-Munich dentro de este paquete:
  20 zonas h6 s3, con delta de servicio de 14.39 puntos porcentuales.
- Las ejecuciones EV h72 de calibración no se incluyen en la tabla principal de
  la memoria; quedan mejor planteadas como calibración futura para una
  publicación más amplia.
