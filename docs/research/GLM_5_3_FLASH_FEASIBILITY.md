# GLM-5.3-Flash on TurboFieldfare: feasibility

Research note, 2026-09-26. Question: can the TurboFieldfare approach (keep the
shared core resident, stream only the selected routed experts from SSD) run
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on a 32 GB
M1 Pro MacBook Pro with a 1 TB SSD?

**Short answer:** the approach fits, but the runtime does not support the
model today. Supporting it means a new model implementation, not a
configuration change. The expected speed on an M1 Pro is about **1-2 tok/s at
4 bits**. Before spending weeks on kernels, measure SSD throughput and the
expert cache hit rate.

## Sources

- `zai-org/GLM-5.3-Flash/config.json` on Hugging Face (architecture).
- Safetensors headers of
  [`Vontra/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP)
  (measured component sizes; MLX affine 4-bit, group 64).
- This repo: [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md),
  [COMMUNITY_BENCHMARKS.md](../COMMUNITY_BENCHMARKS.md),
  [experiment summaries](../experiments/summaries/).
- The z.ai blog and OpenRouter pages did not render any architecture details,
  so they are not used here.

## Architecture compared with Gemma 4 26B-A4B

| | Gemma 4 26B-A4B (current target) | GLM-5.3-Flash |
|---|---|---|
| Total parameters | 26B | ~321B (FP8 E4M3 release) |
| Layers | 30 | 45 (+1 MTP layer) |
| Hidden size | 2816 | 4096 |
| Attention | 25 sliding-window + 5 full, standard KV | 33 linear-attention (KDA) + 11 DeepSeek sparse attention (MLA, top-2048 indexer) |
| Experts | 128 per layer, top-8 | 288 per layer, top-8, +1 shared; first 3 layers dense |
| Router | softmax, 8-bit | sigmoid with `noaux_tc` bias, routed scale 2.5 |
| Residual path | standard, learned layer scalar | mHC hyper-connections (4 streams, Sinkhorn mixing) |
| Output | tied embedding, final softcap 30 | separate LM head |
| Vision | optional image pack | 24-layer vision encoder |

## Measured sizes (MLX 4-bit)

| Part | Size | Placement |
|---|---|---|
| Routed experts (42 layers × 288, 14.16 MB each) | **171.2 GB** | SSD, streamed |
| Attention | 3.5 GB | resident |
| Shared experts, dense MLP, routers | 0.9 GB | resident |
| Embedding + LM head | 0.7 GB | resident |
| **Text core** | **~5.1 GB** | resident |
| MTP layer | 4.2 GB | optional |
| Vision | 1.1 GB | optional |
| Total checkpoint | 181.7 GB | |

KV state is small. The 33 KDA layers keep a fixed recurrent state of about
140 MB in total. The 11 MLA layers store about 11 KB per token. A 32K context
needs well under 1 GB.

## Memory budget on 32 GB

| Item | Size |
|---|---|
| macOS + app | ~5 GB |
| Text core | ~5 GB |
| KV / recurrent state | ~0.5 GB |
| **Expert cache** | **~18-20 GB ≈ 1,300 experts ≈ 31 of 288 per layer (~11%)** |

Gemma's default caches 16 of 128 experts per layer (12.5%), so the cached
share is similar.

## Per-token I/O

| | Gemma 4 | GLM-5.3-Flash |
|---|---|---|
| Experts touched per token | 240 | 336 |
| Expert size | 3.4 MB | 14.2 MB |
| Bytes per token, cold | ~0.8 GB | ~4.8 GB |
| Expert file fits in the OS file cache? | mostly (12 GB file) | no (171 GB file) |

At about 5 GB/s parallel read on a 1 TB M1 Pro SSD:

| Hit rate | Miss bytes per token | Read time |
|---|---|---|
| 30% | 3.3 GB | ~0.65 s |
| 50% | 2.4 GB | ~0.5 s |
| 70% | 1.4 GB | ~0.3 s |

The hit rate has not been measured. The aux-loss-free balanced router spreads
tokens evenly across experts, which suggests weaker locality than Gemma's.

## Compute ceiling on M1 Pro

Gemma 4 decodes at about 9.5 tok/s on a 16 GB M1 Pro (issue #37). GLM does
roughly 4-5× more work per token: it reads about 10 GB of weights versus about
2 GB, has 45 layers versus 30, and uses heavier attention. A rough ceiling
with every expert resident is **about 2-3 tok/s**. This is extrapolated, not
measured.

Read and compute overlap only partly, so the combined estimate is **about
1-2 tok/s at 4 bits**. For comparison, `mmap`-based loading (llama.cpp) was
about 8× slower than parallel `pread` in this repo's IO-01 experiment.

## Levers

- **2.5-3-bit experts:** smaller reads and more experts in cache. This might
  reach 2-3 tok/s, at which point compute becomes the limit. Quality drops
  somewhat.
- **MTP self-speculation:** the MTP layer drafts the next token so two tokens
  are verified per step. Adjacent tokens often pick the same experts, so
  reads are shared. Add this after the base path works.
- **Disk:** 171 GB of experts against about 205 GB free. It fits, with little
  margin.

## What is reusable and what is new

**Reusable:** the parallel `pread` expert streamer with LFU slots, the
range-based installer and repacker (layout only, no requantization), the
affine 4-bit matmul and MoE kernels (new shapes only), and the decode
service, server, and app.

**New:**

1. KDA linear-attention kernels (gated delta rule, short conv, chunked prefill).
2. MLA with the DeepSeek sparse-attention indexer (top-2048, kpool compression).
3. mHC hyper-connections, which change every layer's residual wiring.
4. The sigmoid router with bias correction and ×2.5 scaling.
5. A new on-disk format and repacker. The `GTurbo*V1` format is Gemma-shaped.
   The source would be an MLX 4-bit build.
6. GLM's tokenizer and chat template.

Items 1 and 2 are most of the work. Reference implementations: transformers
`glm5_next`, and the MLX implementation behind the existing MLX builds.

## Next steps (cheap gates before kernel work)

1. **SSD simulator (about 1 hour):** issue parallel `pread` calls of random
   14.2 MB blobs, 336 lookups per token, over a range of hit rates. Use a test
   file of 40 GB or more, larger than RAM, so the page cache cannot hide
   misses. This follows IO-01.
2. **Hit-rate replay:** capture per-token routing traces from an existing
   build (for example MLX 4-bit on a large-memory machine) and replay them
   against about 31 LFU slots per layer. This follows CACHE-03.

If the simulator shows less than about 2 tok/s at realistic hit rates, the
4-bit port is not worth it on an M1 Pro.
