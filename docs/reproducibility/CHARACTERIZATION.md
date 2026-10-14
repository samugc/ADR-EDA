# Runtime characterization

The public Docker environment was checked against the historical ADR-EDA
experiment environment before repository cleanup.

## Functional characterization

The two environments were compared on:

- problem: `df1_t10`
- seed: `0`
- horizon: `60`
- population size: `40`
- decision variables: `8`
- generations per step: `1`
- probabilistic model: `SPBN`

All recorded outputs matched except wall-clock runtime.

The characterization run exercised:

- 5 detected environmental changes
- predictive response
- continuity response
- diversity response
- ADR response-weight computation
- reaction quality safeguard enabled; no reactions reverted in this run
- SPBN model updates

Wall-clock runtime is not treated as a reproducibility target because it
depends on host load, container state and architecture emulation.

Recorded coverage limits: `reaction_front_ls_samples = 0` (no actual
Pareto-front local-search sampling) and `reaction_quality_reverted_count = 0`.

<details>
<summary>Technical reference: validated source identities</summary>

## Source identity

The reconstructed environment uses the same ADR-EDA `DynamicEDA`
implementation and the same benchmark runner used in the historical
experiment workspace.

`DynamicEDA` SHA-256:

```text
552e4f4856e7d14a49759b7753c27b0881bb4272b9468bee1b6974aab8a45347
```

Benchmark runner SHA-256:

```text
8dd8046475a1b2570a1943e75923ac9407f187dc6ce9eda29a18a2c757910187
```


</details>
