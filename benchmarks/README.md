# Benchmarks

Benchmarks ramp up in two phases:

1. **Microbenchmarks with dummy tensor workloads.** Drive the allocator with
   synthetic reservation traces. Goal: measure raw allocator throughput,
   fragmentation under adversarial sequences, and bandwidth utilization on
   cache reads. No model dependency, no tokenizer, no logits. Fast iteration.
2. **Toy hybrid models.** 130M to 300M parameter models that fit comfortably
   on a single RTX 3060 12GB. Goal: validate that microbenchmark wins survive
   real layer wiring. Toy models are picked so a full run completes in
   minutes, not hours.

## Target hardware

The reference machine is a single RTX 3060 with 12GB of VRAM. The benchmarks
must run end to end on this card without offloading. Larger cards (24GB,
40GB) are welcome but not required. CI runs CPU only and skips anything
marked with the `gpu` pytest marker.

## What we measure

- Allocator throughput in reservations per second.
- Fragmentation ratio after a long mixed-shape trace.
- Cache bandwidth utilization during a read-dominated workload.
- Peak VRAM occupancy under a target batch size.
- Allocator overhead as a fraction of total step time.

## What we do not measure here

- End-to-end model perplexity. Quality belongs in a separate evaluation
  harness once a concrete allocator lands.
- Cross-card scaling. Single-GPU only for the first milestone.

## Running

The first concrete benchmark is `dummy_cache_workload.py`. It is stubbed out
until the first allocator lands. See the file for the TODO list.
