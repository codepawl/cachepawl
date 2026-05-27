# vLLM Path C Metrics Table

This table records the original `4096 / 0.7 / 1` baseline cell. The completed
4-cell advisory matrix is recorded in `matrix_table.md` and `matrix_table.csv`.

| Metric | Value |
| --- | ---: |
| `vanilla_reserved_bytes` | `2910781440` |
| `vanilla_useful_bytes` | `1679258112` |
| `cachepawl_proposed_reserved_bytes` | `1679258112` |
| `estimated_savings_bytes` | `1231523328` |
| `overestimation_ratio` | `1.7333734577189286` |
| `wasted_fraction` | `0.4230902777777778` |
| `cache_group_count` | `7` |
| `cache_tensor_count` | `9` |
| `layer_count` | `63` |
| `num_blocks` | `329` |

## Contract Summary

| Contract | Status | Evidence |
| --- | --- | --- |
| Planner-stage replay matched runtime scheduler config | Observed | `planner_matches_runtime_scheduler=true` |
| Runtime changed during replay | Not observed | `runtime_changed_during_replay=false` |
| Request-to-block assignment | Observed | live request block ids `[1, 2, 3, 4, 5, 6, 7]` |
| Worker tensor layout | Observed | 32 worker tensor summaries; first shape `[2, 329, 48, 32, 160]` |
| Attention block-table/view | Observed | 21 block-table tensor metadata summaries |
| Attention metadata builders | Observed | 7 attention groups |
| Mamba state-index contract | Blocked | `mamba_state_idx` reachable but empty for live request |
| Mamba state tensor contract | Blocked | no Mamba state tensors safely reachable by stable runtime attributes |
