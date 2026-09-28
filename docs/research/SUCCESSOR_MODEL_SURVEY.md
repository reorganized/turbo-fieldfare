# Successor models for a 32 GB M1 Pro: survey

Research note, 2026-09-27. Continues
[GLM-5.3-Flash feasibility](GLM_5_3_FLASH_FEASIBILITY.md), which concluded that
GLM-5.3-Flash fits the TurboFieldfare memory model but lands at about
**1-2 tok/s** and needs KDA linear attention, sparse MLA, and mHC kernels.
This survey asks the complementary question: which shipping open models are
more capable than Gemma 4 26B-A4B **and** closer to the runtime's shape?

**Short answer:** two candidates dominate the current ≤40B MoE class:

1. **IFM K2-Horizon-MoVA-36B-A4B** — 36B total, 4B active, ~21 GB at 4-bit.
   Standard softmax GQA attention plus routed value experts. Strongest
   benchmarks in its class; the least new kernel work per unit of quality.
2. **Qwen3.6-35B-A3B** — 35B total, 3B active, ~19 GB at 4-bit. Its entire
   expert pool fits in a 32 GB machine's cache (compute-bound after warmup),
   and its ecosystem is the largest of any small MoE. But it uses gated
   delta-rule linear attention, which is the single most expensive kernel
   item in the GLM writeup.

Most of the rest of the 2026 small-MoE field moved to MLA, linear attention,
or Mamba hybrids and inherits GLM-5.3-Flash's kernel problem.

## Fit criteria

Derived from the runtime and the GLM writeup:

- Routed experts are independent gate/up/down FFNs streamed per token.
- Standard softmax attention with GQA. Partial RoPE, separate rope thetas, a
  full-attention layer mask, and a tied-embedding flag already exist in
  `ArchConfig`; MLA, linear attention, SSMs, and hyper-connections do not.
- MLX affine quantization (group 64) so the repacker copies values unchanged.
- Resident core + expert cache + KV fit in 32 GB with a large expert cache.
- Quality ahead of Gemma 4 31B-it, the strongest Gemma 4 in recent
  comparison tables (26B-A4B is the MoE sibling TurboFieldfare runs today).

## The 2026 small-MoE field

Attention type decides port cost. Surveyed via Hugging Face configs,
September 2026:

| Model | Total/active | Attention | MoE | Verdict |
|---|---|---|---|---|
| **K2-Horizon-MoVA-36B-A4B** | 36B / 4B | softmax GQA + routed value experts | 100 × top-8 + shared | **Best fit** |
| **Qwen3.6-35B-A3B** | 35B / 3B | gated delta-rule (30 of 40 layers) + GQA | 256 × top-8 + shared | Runner-up; GDN kernels |
| Qwen3.8-Flash-Next (180B) | 180B / ~A3B-class | GDN + sparse-attention indexer + ngram | 512 × top-10 | Too much new work |
| occamy-1.0, Tiel-Coder, moziAI 35B-A3B | 35B / 3B | Qwen3.6 derivatives | — | Same GDN cost |
| GLM-4.7-Flash | 31B / ~3B | MLA (q/kv LoRA ranks) | 64 × top-4 | MLA kernels |
| Xing4.0-29B-A4B | 29B / 4B | MLA + mHC hyper-connections | 64 × top-4 | MLA + mHC |
| Mistral Small 4 (119B) | 119B / ~A13B | MLA | switch-MLP | Active too large |
| MiMo-V2.6-Flash (159B) | 159B / ~10B | MLA-family, SWA-128 | 256 × top-8 | Active too large; FP8 native |
| Nemotron 3.5 Lightning 30B-A3B | 30B / 3B | Mamba2 hybrid (`nemotron_h`) | 128 × top-6 | SSM kernels |
| Nemotron 3 Super / Ultra | 120B / 12B, 550B / 55B | hybrid | — | Active too large |
| G9v3-39A5B | 39B / 5B | softmax GQA + output gate | 320 × top-32 | Fits, but benchmarks trail Gemma 4 31B |
| Muse Glimmer-30B | 30B / 30B dense | softmax GQA | — | Dense: ~2-3 tok/s ceiling on M1 Pro |
| Llama 4 Scout / Maverick | 109B / 17B, 400B / 51B | iRoPE GQA | 16/128 × top-1 | Active too large; 2025 generation |
| gpt-oss-120b / 20b | 117B / 5B, 21B / 3.6B | GQA + sinks, SWA | 128 × top-4 | 2025 generation; MXFP4 native |
| GLM-5.3-Flash | 321B / ~4B | KDA + sparse MLA + mHC | 288 × top-8 | See [GLM doc](GLM_5_3_FLASH_FEASIBILITY.md); 1-2 tok/s |

No larger Gemma 4 MoE exists (the family is 12B, 26B-A4B, 31B dense), so
there is no same-architecture step-up.

## Quality, from the K2-Horizon card

The only recent table that puts these models together (open weights,
reasoning mode, scores in %, source: IFM model card, which cites Artificial
Analysis):

| Benchmark | K2-Horizon-MoVA 36B-A4B | Nemotron 3 Ultra 550B | Nemotron 3 Super 120B | G9v3-39A5B | Qwen3.6-35B-A3B | Muse Glimmer-30B | Gemma 4 31B-it |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agentic tool use (tau3-Banking) | **26.8** | 14.2 | 10.3 | 22.1 | 9.3 | 23.5 | 14.8 |
| Terminal-Bench 2.1 | **58.6** | 53.9 | 38.6 | 32.6 | 44.9 | 51.7 | 43.4 |
| SciCode | 38.9 | **39.9** | 36.0 | 34.0 | 35.8 | 43.6 | 43.4 |
| Humanity's Last Exam | 25.2 | **28.4** | 20.8 | 17.5 | 22.2 | 22.0 | 23.6 |
| GPQA Diamond | 80.8 | **86.7** | 80.0 | 80.5 | 84.1 | 83.5 | 85.7 |
| CritPt | 2.1 | **3.1** | 3.1 | 0.3 | 0.3 | 2.6 | 1.4 |
| AA-LCR (long context) | 66.3 | 71.0 | 60.3 | 62.0 | 66.7 | **80.0** | 68.3 |
| AA-Omniscience accuracy | 18.8 | 22.6 | 24.3 | 14.9 | 18.8 | **27.0** | 20.0 |
| AA-Omniscience non-halluc. | 69.2 | 70.3 | 13.0 | **87.0** | 49.5 | 18.1 | 15.0 |

K2-Horizon-MoVA-36B-A4B beats Gemma 4 31B-it on six of nine rows, including
agentic tool use (26.8 vs 14.8), Terminal-Bench (58.6 vs 43.4), and
non-hallucination (69.2 vs 15.0), while running 4B active. Qwen3.6-35B-A3B is
mixed against Gemma 4 31B-it: ahead on Terminal-Bench and non-hallucination,
behind on GPQA, HLE, and AA-LCR. Note the table lacks Gemma 4 26B-A4B itself;
treat "beats 31B-it" as the conservative bar for "more powerful than Gemma 4".

## K2-Horizon-MoVA-36B-A4B in detail

Architecture (from `config.json` and the repo's `modeling_k2_horizon.py`,
which is generated from the Qwen3-MoE transformers modular):

- 48 layers, hidden 2560, 512K context. Text-only; Apache-2.0; no MTP.
- 45 MoE layers with **100 routed experts, top-8**, sigmoid router with
  noaux_tc-style bias applied to selection scores only, normalized weights
  ×2.5, one shared expert; first 3 layers dense.
- **MoVA attention**: 45 of 48 layers replace `v_proj` with **64 routed value
  experts, top-4 per token** (same sigmoid+bias router pattern), each a
  2560→1024 linear with SiLU, whose weighted sum *is* the token's V. Q/K are
  ordinary GQA projections (32 Q heads, 8 KV heads, head dim 128, full RoPE,
  θ=1e7). The mixed V is cached as plain V — the KV cache is ordinary GQA
  shape. A softplus (β=ln 2) output gate and standard o_proj follow.
- 3 plain-attention dense layers at the bottom; T5-style RMSNorm with 2
  groups; untied LM head; no softcap; no learned layer scalar; no mHC.

Measured component sizes at MLX affine 4-bit (group 64, ≈4.5 bits/param;
cross-check: 37B × 0.5625 ≈ 20.8 GB):

| Part | Size | Placement |
|---|---:|---|
| FFN routed experts (45 layers × 100, 3.32 MB each) | **14.93 GB** | SSD, streamed |
| MoVA value experts (45 × 64, 1.47 MB each) | **4.25 GB** | SSD, streamed |
| Attention (q/k/o/gate) | ~0.9 GB | resident |
| Shared experts + dense MLPs | ~0.23 GB | resident |
| Embedding + LM head (untied, 250,624 vocab) | ~0.72 GB | resident |
| Routers, norms, biases | ~0.04 GB | resident |
| **Text core** | **~1.9 GB** | resident |
| **Total at 4-bit** | **~21.1 GB** | |

Per token: 8 FFN experts + 4 value experts per MoE layer →
**~1.46 GB cold** (1.19 FFN + 0.27 value), vs Gemma's ~0.8 GB and GLM-5.3's
~4.8 GB. KV is 184 KB/token FP16 (8 heads × 128 × 2 × 2 B × 48 layers):
3.0 GB at 16K, 6.0 GB at 32K.

### 32 GB budget

| Item | 16K context | 32K context |
|---|---:|---:|
| macOS + app | ~5 GB | ~5 GB |
| Resident core | ~1.9 GB | ~1.9 GB |
| FP16 KV | ~3.0 GB | ~6.0 GB |
| **Expert cache** | **~21 GB** | **~18 GB** |
| Expert pools (FFN + value) | 19.2 GB | 19.2 GB |

The two expert pools total 19.2 GB — **the entire pool fits in cache** on a
32 GB machine at 16K FP16 (or 32K with 8-bit KV). After warmup, decode is
compute-bound; cold misses mostly affect the first chunks of a session and
long-tail experts. Disk cost (~21 GB) is trivial next to GLM-5.3's 171 GB.

Expected speed: Gemma 4 26B-A4B decodes at ~9.5 tok/s on a 16 GB M1 Pro
(issue #37) with ~2.8-3.4B active parameters. K2-Horizon's per-token tensor
count is ~4.5-5B (official "4B"), so a warm-cache extrapolation is roughly
**6-9 tok/s**, modulo the value-expert mixing overhead in attention. This is
an extrapolation, not a measurement.

### Port work

Reusable: parallel `pread` expert streamer with LFU slots (value experts are
just a second, smaller blob class), range-based installer/repacker, affine
4-bit/8-bit kernels (new shapes only), decode service, server, app.

New, in rough size order:

1. **Value-expert mixing path** in attention: a second router + top-4 + SiLU
   + weighted sum producing V before the attention kernel. The reference
   implementation is ~200 lines; the blob streaming reuses the MoE machinery.
2. **Sigmoid router with selection-only bias** for both router kinds, ×2.5
   scaling. Small kernels; selection semantics must match `calc_router_weights`
   (bias added after sigmoid, applied to selection only).
3. **Softplus output gate** (β=ln 2) fused into the attention epilogue.
4. T5-style **grouped RMSNorm** (2 groups); 3 dense layers; untied LM head
   (the `tieWordEmbeddings` flag exists but the head path assumes tied today).
5. A second expert class in the `.gturbo` layout (`v_experts` alongside
   `mlp.experts`), a new `arch` value, and the K2 tokenizer/chat template.
6. Producing the **MLX affine 4-bit source**. Only BF16 (74 GB), GGUF, NVFP4,
   and EXL3 exist today. Quantization is a one-time job on a ≥96 GB machine
   (or shard-wise); there is no MLX build to copy values from yet.

Risks: two-week-old model, single final checkpoint, technical report still
pending (promised end of September 2026), and MoVA is novel enough that a
subtle mismatch (e.g., router numerics — the card already flags an
`xllm_source_router_gemm_partitions` serving override) would be hard to
debug against references. Mitigation: the GGUF and NVFP4 builds and the
transformers implementation give correctness targets.

## Qwen3.6-35B-A3B (runner-up)

Architecture: 40 layers — 30 gated delta-rule linear-attention + 10
full-attention GQA (16 Q / 2 KV heads, head dim 256, partial RoPE 0.25) —
256 experts top-8 (intermediate 512) + 1 shared, untied head, 262K context,
built-in MTP layer, vision tower in the same repo (the existing companion-pack
pattern applies).

Sizes at 4-bit (computed the same way):

| Part | Size |
|---|---:|
| FFN routed experts (40 × 256 blobs @ 1.77 MB) | **18.1 GB** |
| Resident core (attention, GDN, shared, embeddings, head) | ~1.2 GB |
| Total text at 4-bit | ~19.3 GB |

Per token: 8 × 40 experts → **~0.57 GB cold** — less than Gemma. The pool
(18.1 GB) plus core plus 32K KV (~0.7 GB across only 10 full-attention
layers, plus ~94 MB GDN state) fits with the **whole pool cached** in a
32 GB machine, so decode is compute-bound at ~3B active — a warm extrapolation
is **10-13 tok/s**, ahead of Gemma.

Ecosystem is the best in class: official FP8, `mlx-community` 4-bit builds
(25k+ downloads), unsloth UD-MLX-4bit, dozens of finetunes, MTP GGUFs, and a
draft-model variant. Quality is a step below K2-Horizon but roughly at the
Gemma 4 31B dense level, and clearly ahead of the 26B-A4B on agentic work.

The blocker is the same item 1 as in the GLM writeup: **gated delta-rule
kernels** (chunked scan + short conv). The scope is smaller than GLM-5.3
(only 30 layers, no KDA chunked-prefill quirks, no mHC, standard GQA on the
other 10 layers), and `mlx-lm` already runs Qwen3-Next/Qwen3.6 on Apple
Silicon, so a working reference exists to port and to diff against.

## Recommendation

Primary: **K2-Horizon-MoVA-36B-A4B**. It is the strongest model in the class,
its attention stays softmax-GQA-shaped, both expert pools reuse the existing
streamer, and everything fits a 32 GB M1 Pro with the whole pool cacheable.
Its main prerequisites are the value-expert kernel path and a one-time MLX
4-bit quantization on a large machine.

Secondary: **Qwen3.6-35B-A3B**, if GDN kernel cost is acceptable — it trades
more new kernels for better economics, an enormous ecosystem, and a ready
MLX source.

Both need the same first gates as the GLM plan, adjusted for size:

1. **SSD simulator** (IO-01 method): 3.32 MB blobs at 360 lookups/token
   (FFN) plus 1.47 MB blobs at 180 lookups/token (value), across hit rates.
   On 32 GB the realistic operating point is a near-full cache, so this gate
   mostly prices the cold start and the 16 GB fallback.
2. **Hit-rate replay** (CACHE-03 method): capture FFN routing from an MLX run
   (Qwen3.6's `mlx-community` build, or K2's GGUF via llama.cpp traces) and
   replay against the slot budget per layer.
3. For the Qwen3.6 path only: benchmark `mlx-lm`'s Qwen3-Next GDN path on an
   M1 Pro before committing to Metal scan kernels.

## Sources

- `IFM/K2-Horizon-MoVA-36B-A4B`: `config.json`, `model.safetensors.index.json`
  (tensor inventory), `modeling_k2_horizon.py` (MoVA semantics, router
  math), and the model card's benchmark table (citing Artificial Analysis).
- `Qwen/Qwen3.6-35B-A3B`: `config.json` (hybrid layer types, MoE shape);
  `mlx-community/Qwen3.6-35B-A3B-4bit` index (20.4 GB total including vision).
- Configs of rejected candidates: `Qwen/Qwen3.8-Flash-Next`,
  `zai-org/GLM-4.7-Flash`, `ai9stars/G9v3-39A5B`,
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
  `mlx-community/Xing4.0-29B-A4B-OptiQ-4bit`,
  `mlx-community/Mistral-Small-4-119B-2603-OptiQ-2bit`,
  `dealignai/MiMo-V2.6-Flash-RL-UNCENSORED`, `Accio-Lab/occamy-1.0`.
- Hugging Face org listings (Qwen, zai-org, google, openai, meta-llama,
  mlx-community), September 2026, for recency and ecosystem checks.
- This repo: [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md),
  [GLM-5.3-Flash feasibility](GLM_5_3_FLASH_FEASIBILITY.md),
  [experiment summaries](../experiments/summaries/).
