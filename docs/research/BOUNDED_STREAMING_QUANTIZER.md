# Bounded per-tensor streaming quantizer

Idea capture, 2026-09-27. Not scheduled. Continues
[SUCCESSOR_MODEL_SURVEY.md](SUCCESSOR_MODEL_SURVEY.md) (K2-Horizon section).

## Problem

TurboFieldfare's repacker copies quantized values unchanged from a pinned MLX
source; it never requantizes. That requires an MLX build to exist. Models
without one — e.g. K2-Horizon-MoVA-36B-A4B today (BF16 74 GB, GGUF, NVFP4,
EXL3, no MLX) — need a quantization pass first. Stock `mlx_lm.convert -q`
holds the whole model in memory (~96 GB for K2's BF16 release), which is
hardware we do not have.

## Idea

A quantizer whose memory bound is one tensor tile, not one model:

- Walk the source safetensors index. Per tensor, read the header, then stream
  the tensor in row tiles (input tile bounded, e.g. ≤256 MiB).
- Affine group-64 quantization is row-local: each group of 64 elements along
  the last dim gets its own min/max → scale/bias, so tiles quantize
  independently and pack immediately into the destination file.
- Peak memory: one input tile + one output tile (≪1 GB regardless of model
  size). Runs on the 32 GB M1 Pro.

## Compatibility with the repo's invariant

- **Option A (preferred):** the quantizer emits a standalone MLX-layout build
  (a normal HF repo tree on disk). The existing range-repacker then consumes
  it exactly as any pinned community build — copying bytes verbatim. The
  quantization is a separate, explicitly pinned tool step; the pin records
  our output's index SHA-256 like the Gemma source pin does.
- **Option B (later):** fuse quantize + repack into one streaming pass.

## Where it could live

- `TurboFieldfareRepackCore` subcommand (Swift + Foundation, matches the
  repo, testable with the model-free test style), or
- a standalone Python script (safetensors + numpy; the math is simple) run
  once, with the output kept as the pinned local source.

## What needs care

- This **creates a new quantization** (RTN affine, the same algorithm mlx
  uses by default) instead of copying upstream numbers. Quality gate:
  perplexity plus logit-diff fixtures against the BF16 reference on held-out
  prompts, reported before the build is pinned.
- Bits mix: mlx builds typically quantize linear layers 4-bit and keep
  embedding/LM head at 8-bit. Decide per model and record it in the manifest.
- Tiling: safetensors tensors are contiguous; tile along dim 0 with group
  alignment along the last dim. Scales/biases accumulate in fp32, cast to
  BF16 once per group.
- Determinism: fixed tile size and fixed reduction order, so the output
  digest is reproducible.

## Open questions

1. Does mlx RTN use plain per-group absmax scale/bias, or a clamped/MSE
   variant? Match it exactly if we want drop-in comparability with community
   builds.
2. Replicate the mlx mixed-precision policy (8-bit embed/head) per model, or
   define our own?
3. Publish the resulting build to a HF repo we own, or keep it local and pin
   the manifest to the local digest?
