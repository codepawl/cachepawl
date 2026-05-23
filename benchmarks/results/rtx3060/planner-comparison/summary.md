| backend | workload | useful_bytes | estimated_bytes | overestimation_ratio | wasted_fraction | virtual_oom | planner_runtime_us |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-style-padded | short-heavy | 2435055616 | 6985613312 | 2.868770 | 0.651418 | false | 0.000 |
| cachepawl-avmp | short-heavy | 2435055616 | 2451046400 | 1.006567 | 0.006524 | false | 0.000 |
| vllm-style-padded | long-heavy | 41967550464 | 165116116992 | 3.934376 | 0.745830 | true | 0.000 |
| cachepawl-avmp | long-heavy | 41967550464 | 41983672320 | 1.000384 | 0.000384 | true | 0.000 |
| vllm-style-padded | mixed | 13517422592 | 51311017984 | 3.795917 | 0.736559 | true | 0.000 |
| cachepawl-avmp | mixed | 13517422592 | 13532397568 | 1.001108 | 0.001107 | true | 0.000 |
