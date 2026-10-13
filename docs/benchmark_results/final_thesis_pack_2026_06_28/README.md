# Paquete benchmark final para la memoria

Este paquete reúne los artefactos consolidados usados por la memoria para la
validación en benchmarks dinámicos. Su objetivo es proporcionar una ruta limpia
de tesis para los informes, tablas y ficheros agregados utilizados en el
Capítulo 4 y en el anexo de reproducibilidad.

Contenido principal:

- `benchmark/analysis/analysis_report.md`: informe agregado de la suite
  benchmark principal.
- `benchmark/analysis/overall_table.csv`: tabla global de resultados.
- `benchmark/analysis/family_table.csv`: resultados agregados por familia.
- `benchmark/per_run_metrics.csv`: métricas por ejecución de la campaña
  principal.
- `benchmark/summary_compact.csv`: resumen compacto de la campaña principal.
- `benchmark_adr_dmoea/REPORT.md`: comparación independiente con ADR-DMOEA.
- `benchmark_adr_dmoea/summary.json`: resumen estructurado de esa comparación.
- `scalability/n20_h20_s2/REPORT.md`: prueba compacta con 20 variables.
- `scalability/n20_h20_s2/summary.json`: resumen estructurado de la prueba
  dimensional.

Los nombres de carpetas se mantienen separados por bloque experimental para
evitar mezclar la campaña principal, la comparación con ADR-DMOEA y la prueba de
sensibilidad dimensional.
