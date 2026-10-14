# Third-party provenance and licensing status

No repository-wide license is asserted. Availability of source and results
is not a grant of unrestricted reuse or redistribution rights.

| Material | Recorded provenance | Status for this release |
| --- | --- | --- |
| EDAspy extension | [EDAspy upstream](https://github.com/VicentePerezSoloviev/EDAspy) plus the supplied patch; version specified in the Docker build | Upstream package metadata identifies MIT; upstream notices and the patch's contribution rights must be considered separately |
| Dynamic comparators | `tests/edas/clean_dynamic_sota_comparison.py` describes reference implementations and author-code-audited Python ports | File-level permissions for ports are not established in this package |
| SGEA | Runner attributes Jiang–Yang (2017) and an audited author-code Python port | Original author-code licensing/derivation needs review |
| Tr-NSGA-II, PFPA, TDA-DMOEA | Runner attributes Jiang et al. (2018), Zhang et al. (2022), Cai et al. (2024) | Paper attribution does not itself establish code licensing |
| AB-DMOEA | Runner attributes Peng et al. (2024), author-code-audited Python port | Original author-code licensing/derivation needs review |
| ADR-DMOEA | Runner attributes Wang et al. (2026) | Implementation provenance and permissions need review |
| Mobility inputs and derived evidence | Munich demand, Citi Bike/Divvy profiles, OSRM-derived distances, final application tables | Raw demand is absent; dataset terms, routing/map attribution and rights in derived artifacts remain to be verified |
| ADR-EDA code and scientific artifacts | Research by Jesús Samuel García Carballo, with thesis/CAEPIA provenance | No blanket license assigned pending author/institution and third-party rights review |

The runner mentions historical `third_party/author_code` sources; that directory
is not part of this release. It must not be interpreted as bundled permission.
The required Madrid routing fixture remains tracked at
`tests/model/results/madrid_osrm_zone_matrix_congestion_2025_01.json`.

Before assigning a repository-wide license, establish the licenses/notices for
upstream code and ports, contribution rights for the extension and application
code, and redistribution/attribution terms for data and derived routing artifacts.
This document records gaps; it does not resolve them or change historical files.
