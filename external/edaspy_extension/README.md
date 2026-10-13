# EDAspy extension

The characterized ADR-EDA core is reconstructed from
[EDAspy upstream](https://github.com/VicentePerezSoloviev/EDAspy) at commit
`04e45cfb9f797cc735dd0f5b265ac03c496100d4` (also recorded in [BASE_COMMIT](BASE_COMMIT)).

[adr_eda_edaspy.patch](adr_eda_edaspy.patch) adds/adapts dynamic EDA execution,
multiobjective selection, SPEDA/DynamicSPEDA integration and small-sample
robustness for probabilistic models. It is required for the characterized
runtime; unpatched upstream is not an equivalent implementation.

The full upstream repository is not vendored: a pinned base and patch preserve
reviewable provenance without duplicating the dependency. The canonical
[Dockerfile](../../docker/Dockerfile.benchmark) clones the base, verifies the
patch, applies it and installs the result automatically during
`./scripts/setup.sh`. No manual checkout is required.

To inspect the patch from the ADR-EDA repository root:

```bash
git apply --stat external/edaspy_extension/adr_eda_edaspy.patch
```

Do not apply it to an arbitrary newer upstream revision. See the
[runtime characterization](../../docs/reproducibility/CHARACTERIZATION.md)
for source identities and [third-party provenance](../../THIRD_PARTY.md)
for licensing scope. Historical development names inside the patch are retained.
