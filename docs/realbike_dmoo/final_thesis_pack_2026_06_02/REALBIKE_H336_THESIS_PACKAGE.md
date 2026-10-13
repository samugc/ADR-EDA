# Paquete de tesis RealBike multiciudad

Este paquete conserva la evidencia RealBike usada en la memoria: Citi Bike y
Divvy con demanda real a nivel de estación, matrices OSRM, predicción
`historical_hour` en línea, calentamiento de 72 horas, horizonte de 336 horas y
EDAspy/SPEDA para las políticas EDA.

## Lectura central

ADR-EDA debe leerse como un controlador adaptativo consciente del coste, que decide cuándo actualizar y cuándo reutilizar el objetivo aprendido. Las políticas que recalculan cada hora pueden ganar servicio moviendo más distancia; la aportación de ADR-EDA es mantener un servicio competitivo reduciendo muchas actualizaciones completas.

## Lectura por sistema

### Divvy

- Utilidad ADR-EDA: 1896.50.
- Servicio ADR-EDA: 0.8960.
- Kilómetros OSRM ADR-EDA: 2238.80.
- Reoptimizaciones/reutilizaciones ADR-EDA: 183.1/152.9.

- Frente a reeq. horario: delta de utilidad 142.45, delta de servicio 0.0012, delta de km -2051.14, delta de reoptimizaciones -152.90.
- Frente a horizonte rodante: delta de utilidad 111.94, delta de servicio -0.0040, delta de km -2462.93, delta de reoptimizaciones -152.90.
- Frente a EDA horaria: delta de utilidad -31.20, delta de servicio -0.0019, delta de km 156.82, delta de reoptimizaciones -152.90.

### Citi Bike

- Utilidad ADR-EDA: 8609.63.
- Servicio ADR-EDA: 0.9956.
- Kilómetros OSRM ADR-EDA: 3569.57.
- Reoptimizaciones/reutilizaciones ADR-EDA: 146.1/189.9.

- Frente a reeq. horario: delta de utilidad 30.88, delta de servicio -0.0010, delta de km -782.29, delta de reoptimizaciones -189.90.
- Frente a horizonte rodante: delta de utilidad 18.08, delta de servicio -0.0015, delta de km -837.83, delta de reoptimizaciones -189.90.
- Frente a EDA horaria: delta de utilidad -8.59, delta de servicio 0.0000, delta de km 120.59, delta de reoptimizaciones -189.90.

## Encaje en la memoria

- Citi Bike y Divvy constituyen la evidencia principal de RealBike en la memoria.
- La formulación más sólida es: ADR-EDA reduce de forma sustancial las reoptimizaciones completas y ofrece una frontera reproducible entre servicio, coste y esfuerzo computacional sobre demanda real de bicicletas compartidas a nivel de estación.

## Archivos

- `realbike_h336_summary_table.csv` / `.tex`: indicadores absolutos por política.
- `realbike_h336_pairwise_deltas.csv` / `.tex`: tabla central de deltas pareados.
- `realbike_h336_service_vs_km.svg` / `.png`: figura de servicio y coste.
- `realbike_h336_reopt_vs_decision_time.svg` / `.png`: figura de esfuerzo computacional.
