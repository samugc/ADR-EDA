# Public release checks

Validation performed on 2026-10-14 before initial Git commit.

| Check | Result |
| --- | --- |
| Shell syntax: setup, small benchmark, full launcher | Passed `bash -n` |
| Both public benchmark help interfaces | Passed |
| Modified collector Python syntax | Passed AST parsing and in-memory compilation; no bytecode written |
| Full-launcher host Docker dispatch | Passed with a recording Docker stub |
| Full-launcher chunk orchestration | Passed with a simulated runner: 112 chunks, injected failure, failure exit status, retry of only the failed chunk, completion skipping, protocol-mismatch refusal |
| Collector on simulated chunks | Passed |
| Docker Compose configuration | Passed; `linux/amd64` preserved |
| Existing Docker image dependency consistency | `pip check`: no broken requirements |
| DynamicEDA identity in existing Docker image | Matched the characterized SHA-256 below |
| Benchmark runner identity | Matched the characterized SHA-256 below |
| Public Markdown relative links | All checked targets exist |
| CFF YAML and essential metadata | Parsed with Ruby YAML; author/repository metadata reviewed; full CFF schema validation not run |
| Private absolute paths | No user-specific absolute paths found in public/source/evidence text during audit |
| Secret patterns | No obvious token, key, private-key or credential assignment matches found; heuristic scan, not a guarantee |
| Caches and large temporary artifacts | No Python bytecode/cache directories found; no files above 40 MB |
| Scientific preservation | No historical code/results deleted or renamed; required Madrid fixture retained |
| Final statistics | Post-change Wilcoxon values reproduced from 120 finite pairs using continuity correction |

Benchmark runner SHA-256:

```text
8dd8046475a1b2570a1943e75923ac9407f187dc6ce9eda29a18a2c757910187
```

DynamicEDA SHA-256:

```text
552e4f4856e7d14a49759b7753c27b0881bb4272b9468bee1b6974aab8a45347
```

The existing image was inspected; no clean image rebuild or scientific campaign
was run during this release pass. The full campaign was not executed. Launcher
tests used simulated outputs stored under ignored `results/release_checks/`.
The earlier runtime characterization remains the evidence for numerical runtime
agreement; the new release checks do not imply fresh end-to-end replication.

Scientific material surfaced: the final benchmark overall/family tables and
per-run metrics, the separate ADR-DMOEA report, final controlled car-sharing
paired deltas, Munich transfer tables and the final 336-hour bike-sharing table.
A Mermaid diagram explains the architecture without copying a thesis screenshot.
All selected evidence already existed in this repository; no external historical
source material needed copying or modification.

Historical result tree names, dated scripts, original report headings, runner
internals and the EDAspy patch are preserved. See `THIRD_PARTY.md` at the root
for unresolved repository licensing, comparator-port rights and dataset/routing
attribution. Raw application inputs are absent; some application workflows
therefore need external data to rerun.

An initial YAML check inside Docker failed because PyYAML is not installed;
YAML parsing was completed using the host Ruby standard library. A first
statistical calculation returned NaN because missing pairs were included;
filtering finite pairs and using continuity correction reproduced the supplied
final values. No dependency changes were made to accommodate these checks.

Git staging audit: 104 files selected; root `results/`, simulated test artifacts,
caches and local environment files are excluded. All three final evidence trees
and the required Madrid routing fixture are included. `git diff --cached --check`
passes for the edited public files. The whole-tree check returns status 2 for
pre-existing whitespace/line-ending warnings in archived CSVs, the validated
EDAspy patch and one historical source comment. Those files were deliberately
not normalized.
