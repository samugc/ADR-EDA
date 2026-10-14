# EDAspy extension

ADR-EDA extends [EDAspy](https://github.com/VicentePerezSoloviev/EDAspy) with
dynamic multiobjective optimization, adaptive responses and SPEDA/SPBN support.

The [extension patch](adr_eda_edaspy.patch) contains the changes used in the
experiments. `./scripts/setup.sh` installs the compatible EDAspy revision and
applies the patch automatically through Docker; no manual installation is needed.

Build details are maintained in the [Dockerfile](../../docker/Dockerfile.benchmark).
See [runtime validation](../../docs/reproducibility/CHARACTERIZATION.md) and
[third-party provenance](../../THIRD_PARTY.md) for technical and licensing details.
