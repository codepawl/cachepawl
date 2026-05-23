# Product Scope

## Project

Cachepawl

## One-Line Description

Hybrid KV and SSM cache allocator for Mamba-Transformer-MoE model inference.

## Target Users

- Researchers evaluating hybrid cache allocation strategies.
- Engineers integrating cache allocators into LLM serving runtimes such as vLLM.
- Future coding agents working on cachepawl milestones.

## Core Problem

Existing KV cache managers assume pure transformer workloads with uniform per-layer cache shapes. Hybrid attention + SSM + MoE models need a shared VRAM allocator that can serve variable-length KV pages and fixed-size SSM state blocks without excessive padding waste or stranded memory.

## Current Goal

Move from research/prototype artifacts toward an end-to-end vLLM integration by establishing a vanilla baseline and implementing the planned AVMP shim.

## In Scope

- Python AVMP allocator prototypes and invariants.
- Benchmark harnesses, reports, and committed research artifacts.
- Triton correctness-oracle work for v2, with production performance deferred to v2.1.
- vLLM integration planning, shim work, and paired baseline/AVMP evidence.

## Out of Scope

- Unrelated serving runtimes unless explicitly planned.
- Product API stabilization beyond the active milestone.
- Weakening tests, benchmarks, or paper claims to force a result.
- Per-allocate Triton production deployment before the v2.1 batched/deferred API work.

## First Useful Demo

A local vLLM run serving the selected hybrid model with captured vanilla baseline metrics, followed by an AVMP-enabled run that preserves generation behavior and does not increase OOMs.

## Success Criteria

- Repo quality gates pass or skipped checks are explicitly justified.
- `.pawl/active/CURRENT.md` always points at the real current sprint and task.
- vLLM integration work produces paired evidence, not unpaired anecdotes.
- Research notes and paper-facing claims stay aligned with measured results.
