# vLLM Planner-Stage Advisory Diff

Status: `planner_stage_advisory_diff_available`

This artifact records a non-mutating Cachepawl proposed planner result computed
beside the vanilla T002 planner-stage output.

Files:

- `manifest.json` — source paths, output names, and non-mutation flags.
- `diff_report.json` — structured planner-stage advisory diff.
- `summary.md` — concise human-readable summary.
- `group_level_diff.json` — optional per-group diff when derivable.

The Cachepawl proposal is not returned to vLLM and does not change vanilla vLLM
behavior.
