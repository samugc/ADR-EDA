# Clean Dynamic SOTA Comparison

## Config
- Fecha: 2026-06-17
- Algoritmos: adr_eda, sgea, trnsga2
- Problemas: dzdt1_step, dzdt3_shocks, df4_t10
- Seeds: 2
- Horizonte: 20
- Poblacion: 20
- Variables: 20
- Generaciones por step: 1

## Algorithms
- `adr_eda`: ADR-EDA v2.1 frozen benchmark-side profile with predictive, continuity and diversity blocks plus contextual front-LS gating. (framework_wrapper, frozen_profile, change=internal_adr_detector_frozen_profile, source=in_repo, target=frozen_profile)
- `sgea`: Jiang-Yang 2017 SGEA with author-code-audited environment selection, steady-state update and centroid-guided change response. (author_code_audited_python_port, author_code_core_port_unified_protocol, change=paper_detection_reevaluate_fraction, source=author_code_audited_port, author_code=author_provided_c_visual_studio_code_extracted_not_directly_executed, available=yes, target=author_code_audited_core_port)
- `trnsga2`: Jiang et al. (2018) Tr-NSGA-II with change-triggered Tr-IPG/TCA, optimized preimage and NSGA-II refinement. (dedicated_module, paper_faithful_core_optimization_preimage, change=oracle_change_notification, source=in_repo, target=paper_faithful_core_optimization_preimage)

## Summary
- `adr_eda`: MIGD=3.6262, MHV=2.9363, postIGD=1.9399, postHV=3.9958, RT=0.0000, F1=0.0000, n=6
- `sgea`: MIGD=5.0868, MHV=2.5343, postIGD=2.8118, postHV=3.5167, RT=0.0000, F1=1.0000, n=6
- `trnsga2`: MIGD=6.3068, MHV=2.3556, postIGD=3.3777, postHV=3.3118, RT=0.0000, F1=1.0000, n=6

## ADR Pairwise
- `improvement_mean` positivo significa que la variante mejora a `adr_eda` segun la direccion de la metrica; `p_variant_better` es el Wilcoxon unilateral en esa direccion.
- `sgea` vs `adr_eda`: n_paired=6
  migd: n=6, improvement_mean=-1.4606, wins/ties/losses=1/0/5, p_variant_better=0.9531
  mhv: n=6, improvement_mean=-0.4020, wins/ties/losses=2/0/4, p_variant_better=0.7812
  post_change_igd: n=6, improvement_mean=-0.8719, wins/ties/losses=2/0/4, p_variant_better=0.7812
  post_change_hv: n=6, improvement_mean=-0.4791, wins/ties/losses=2/0/4, p_variant_better=0.8906
  recovery_time: n=6, improvement_mean=0.0000, wins/ties/losses=0/6/0, p_variant_better=1
- `trnsga2` vs `adr_eda`: n_paired=6
  migd: n=6, improvement_mean=-2.6807, wins/ties/losses=1/0/5, p_variant_better=0.9531
  mhv: n=6, improvement_mean=-0.5807, wins/ties/losses=2/1/3, p_variant_better=0.8438
  post_change_igd: n=6, improvement_mean=-1.4377, wins/ties/losses=1/0/5, p_variant_better=0.9688
  post_change_hv: n=6, improvement_mean=-0.6840, wins/ties/losses=2/1/3, p_variant_better=0.8438
  recovery_time: n=6, improvement_mean=0.0000, wins/ties/losses=0/6/0, p_variant_better=1

## Notes
- `reference_impl`: implementación limpia en este archivo.
- `dedicated_module`: baseline dedicado implementado en este runner unificado.
- `author_code_audited_python_port`: port Python integrado en el protocolo unificado y auditado contra código oficial localizado; no implica ejecución directa del binario/script original.
- `framework_wrapper`: comparación limpia dentro de tu backbone DynamicEDA, pero no reimplementación paper-faithful completa desde cero.
- `frozen_profile`: perfil archivado y congelado para ADR-EDA v2.1 en benchmark-side.
- `provisional_*`: implementación útil para cribado, pero no debe presentarse como reproducción fiel del paper.
- `precision`/`recall`/`f1` se reportan como diagnóstico de detección, pero no entran en el contraste pareado porque varios baselines reciben notificación/oráculo de cambio mientras ADR-EDA usa su detector interno congelado.
