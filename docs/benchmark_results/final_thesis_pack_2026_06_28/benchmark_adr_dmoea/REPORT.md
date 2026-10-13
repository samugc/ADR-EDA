# Clean Dynamic SOTA Comparison

## Config
- Fecha: 2026-06-17
- Algoritmos: adr_dmoea
- Problemas: dzdt1_step, dzdt1_sine, dzdt2_shape, dzdt3_shocks, fda1_lite, fda2_lite, dconstrained_zdt1, ddtlz2_3obj, df1_t10, df2_t10, df3_t10, df4_t10, df5_t10, df6_t10
- Seeds: 15
- Horizonte: 60
- Poblacion: 40
- Variables: 8
- Generaciones por step: 1

## Algorithms
- `adr_dmoea`: Wang et al. 2026 ADR-DMOEA with subpopulation-level ADRM, static-optimization-distance feedback and MOEA/D-DE refinement. (dedicated_module, paper_core_adaptive_dynamic_response_candidate, change=oracle_change_notification_plus_adaptive_response, source=in_repo, target=paper_core_adaptive_dynamic_response_candidate)

## Summary
- `adr_dmoea`: MIGD=1.0066, MHV=4.1112, postIGD=1.3915, postHV=3.5148, RT=5.8511, F1=1.0000, n=210

## ADR Pairwise
- `improvement_mean` positivo significa que la variante mejora a `adr_eda` segun la direccion de la metrica; `p_variant_better` es el Wilcoxon unilateral en esa direccion.
- `adr_dmoea` vs `adr_eda`: n_paired=0

## Notes
- `reference_impl`: implementación limpia en este archivo.
- `dedicated_module`: baseline dedicado implementado en este runner unificado.
- `author_code_audited_python_port`: port Python integrado en el protocolo unificado y auditado contra código oficial localizado; no implica ejecución directa del binario/script original.
- `framework_wrapper`: comparación limpia dentro de tu backbone DynamicEDA, pero no reimplementación paper-faithful completa desde cero.
- `frozen_profile`: perfil archivado y congelado para ADR-EDA v2.1 en benchmark-side.
- `provisional_*`: implementación útil para cribado, pero no debe presentarse como reproducción fiel del paper.
- `precision`/`recall`/`f1` se reportan como diagnóstico de detección, pero no entran en el contraste pareado porque varios baselines reciben notificación/oráculo de cambio mientras ADR-EDA usa su detector interno congelado.
