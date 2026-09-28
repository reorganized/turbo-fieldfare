# Qwen3.6-35B-A3B on TurboFieldfare: port plan

Planning note, 2026-09-27. Continues
[SUCCESSOR_MODEL_SURVEY.md](SUCCESSOR_MODEL_SURVEY.md) and
[GLM-5.3-Flash feasibility](GLM_5_3_FLASH_FEASIBILITY.md). Target machine:
32 GB M1 Pro, 1 TB SSD. Scope: text-only decode + prefill parity; MTP and
vision are later phases.

**Status: no code implemented yet.** This note records the verified source
facts, the code inventory, and the phase plan. The first implementation
session should execute Phase 1 exactly as specified here.

## Handoff (for the next agent)

State as of 2026-09-27: Phase 1 items 1-6 are implemented as **uncommitted
working-tree changes on instance-c** (`/home/opencode/turbo-fieldfare`; 10
modified sources + 1 test tweak, plus the four untracked files in
`docs/research/`). Nothing has been built or tested yet; no Mac machine has
the changes.

Handoff checklist for an agent working directly on the Mac checkout:

1. Get the changes onto the Mac. The Phase 1 work is committed on
   instance-c as `8322796` ("Add Qwen3.6-35B-A3B repack source and manifest
   arch fields (Phase 1)", parent `37fd33f`). Preferred: fetch/apply that
   commit (it contains the research docs too). Fallback: the same changes as
   a patch via `git diff 37fd33f..8322796`. The Mac checkout must be at
   `37fd33f`; if not, port the edits by hand from the checklist below.
2. `swift build` — expect success; report any compile errors verbatim.
3. `Scripts/test.sh` — the existing suite must stay green (all changes are
   additive; Gemma behavior is untouched by design).
4. Write Phase 1 item 7: a synthetic Qwen snapshot test through the planner
   (config with `layer_types` containing `linear_attention`,
   `mlp.switch_mlp.{gate,up,down}_proj` names, 40 layers × 256 experts,
   8-bit `mlp.gate` + `shared_expert_gate` overrides) plus a format
   round-trip test for the new optional manifest fields, and a test that
   legacy (nil-field) manifests still validate.
5. Smoke-check the CLI parsing without downloading:
   `swift run TurboFieldfareRepack --help` should list the sources
   (`gemma4`, `qwen36`).
6. Optionally start the real repack
   (`swift run -c release TurboFieldfareRepack --source qwen36 --output
   scratch/qwen36.gturbo`) — a ~20 GB download needing ~22 GB free; not
   required to validate Phase 1.

Tooling note (learned 2026-09-27): bridging from instance-c works with flat
`ssh makars-macbook-pro '<command>'` calls — exactly one ssh per command.
Nested ssh (running ssh on the Mac to reach the Mac's public IP) fails with
"Host key verification failed"; earlier timeouts were the Mac being
unavailable. If working directly on the Mac, none of this applies.

## Verified source pin

| Field | Value |
|---|---|
| Repo | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Revision | `38740b847e4cb78f352aba30aa41c76e08e6eb46` |
| Index SHA-256 | `0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea` |
| Shards | 4, total 20.40 GB (includes vision tower; repacker skips it like Gemma's) |
| MTP weights | **not present** in this build (clean; MTP needs a separate source later) |
| Quantization | MLX affine group 64: 4-bit everywhere, **8-bit for the 40 `mlp.gate` routers and 40 `mlp.shared_expert_gate`** (80 overrides) |

Packing: 4-bit = 8 nibbles/U32 (embed `[248320, 256]` = 2048 in-dim), 8-bit =
4 bytes/U32 (`shared_expert_gate [1, 512]` = 2048 in-dim, scales `[1, 32]` =
32 groups of 64).

Verified tensor shapes (from shard 1 safetensors header):

| Tensor | Shape | Notes |
|---|---|---|
| `embed_tokens.weight` | `[248320, 256]` U32 | 4-bit |
| `lm_head.{weight,scales,biases}` | separate | untied head |
| `self_attn.q_proj.weight` (full layers) | `[8192, 256]` U32 | **2× heads: output gate fused into q_proj** (16 gate + 16 head blocks of 256) |
| `self_attn.{k,v,o}_proj` | k/v `[512,·]`, o `[2048, 4096]` | 2 KV heads × 256; real V (no K==V trick) |
| `self_attn.{q,k}_norm.weight` | `[256]` | full layers only |
| `linear_attn.in_proj_qkv` | `[8192, 256]` U32 | q 2048 + k 2048 + v 4096 (16 key heads, 32 value heads, dim 128) |
| `linear_attn.in_proj_z` | `[4096, 256]` U32 | sigmoid output gate, 32 × 128 |
| `linear_attn.in_proj_a`, `in_proj_b` | `[32, 256]` U32 each | low-rank dt / beta projections |
| `linear_attn.A_log`, `dt_bias` | `[32]` BF16 | per value head |
| `linear_attn.conv1d.weight` | `[8192, 4, 1]` BF16 | depthwise, kernel 4, over qkv channels |
| `linear_attn.out_proj` | `[2048, 512]` U32 | 4096 → 2048 |
| `mlp.gate.{weight,scales,biases}` | `[256, 512]` U32 | **8-bit router** |
| `mlp.switch_mlp.{gate,up,down}_proj` | gate/up `[256, 512, 256]`, down `[256, 2048, 64]` U32 | **pre-stacked rank-3, 256 experts** — same fused-expert layout as Gemma's `switch_glu`, maps directly onto per-layer blob machinery |
| `mlp.shared_expert.{gate,up,down}_proj` | 4-bit, intermediate 512 | resident |
| `mlp.shared_expert_gate` | `[1, 512]` U32, 8-bit | 2048→1 scalar gate on the shared branch |

All 40 layers are MoE (40 of every `switch_mlp` tensor in the index); layers
0-2 and every 4th thereafter are `self_attn` (10 total), the rest
`linear_attn` (30).

## Architecture deltas vs Gemma 4 26B-A4B

| | Gemma 4 26B-A4B (today) | Qwen3.6-35B-A3B |
|---|---|---|
| Layers | 30 (25 SWA + 5 full) | 40 (30 gated-delta-rule linear + 10 full GQA) |
| Full attention | 16 Q / 8 KV (SWA) & 2 KV (full), headDim 256/512, K==V, scale 1.0 | 16 Q / 2 KV, headDim 256, **separate V, scale 1/16, sigmoid output gate fused in q_proj, q/k norm** |
| Linear attention | — | Gated delta rule + conv1d(4); state 32×128×128 fp32 (~2 MB/layer, ~63 MB total) + conv window |
| Router | softmax over top-8 × per-expert scale, input pre-scale | softmax over all 256 → top-8 → renormalize; no input scale, no per-expert scale |
| Shared FFN | dense branch, added unweighted | 4-bit FFN + **sigmoid `shared_expert_gate`** on its output |
| Norms | RMSNorm; pre/post-FFN sandwich; learned layer scalar | RMSNorm eps 1e-6 only; no sandwich, no layer scalar |
| Head | tied embedding, softcap 30 | **untied lm_head**, no softcap |
| Activation | gelu_pytorch_tanh | silu |
| Embedding scale | × √hidden | none |
| Expert cache | 128 experts/layer, 3.35 MB blob, 16 slots default | 256 experts/layer, ~1.73 MB blob → **whole pool (18.1 GB) fits in a 32 GB cache** |
| Vocab | 262144 | 248320 |
| RoPE | θ 1e4/1e6, partial 0.25, NeoX | θ 1e7, partial 0.25 (mrope config reduces to standard RoPE for text — verify against mlx-lm in Phase 2) |

## What is already parameterized (reusable as-is)

- Format: resident index (free-form names), packed-expert layout validator
  (offsets/stride/expert mapping generic; expert stride is derived + page
  aligned), vision format (skip), path validation.
- Repacker: range copy planner, snapshot loader, tokenizer sidecar copy
  (`tokenizer.json` etc. all ship with the source), receipt/audit writers,
  `switch_glu`→`switch_mlp` is a one-pattern change (see below).
- Runtime: `PreadExpertStreamer` (slot count + stride from layout),
  resident buffers, receipt validation, scratch sizing from ArchConfig,
  prefill tile scheduler (top-8 ≤ slot budget), MPP prefill QMM (runtime
  M/N/K), dequant kernels (group 64 affine, generic fallbacks for new
  shapes), sampler top-64 fast path (vocab-generic), RoPE kernel (runtime
  theta + rotated pairs).

## The seven work fronts (code inventory)

Full file:line inventory was compiled 2026-09-27 (session notes); the
load-bearing items:

1. **Linear attention is a new execution paradigm.** `KVCacheManager.LayerKind`
   (`KVCacheManager.swift:8`), the per-layer graph in `RealForwardRunner`,
   `validateRuntimeSchema` (`Model.swift:508-660`), `ArchInfo.layer_types`
   (`ArchInfo.swift:44-45` — currently collapses non-full to sliding), and a
   new recurrent-state store (fp32 delta-rule state + conv window) all assume
   attention-with-KV. 30 of 40 layers hit this.
2. **Router semantics.** Gemma's input pre-scale + softmax-over-top-8 +
   per-expert scale are baked into `router_gemv_gemma4_r4` /
   `router_topk_select_k8` (`moe.metal:78-190`),
   `prefill_router_gemma4_block` (`prefill.metal:478`),
   `encodeRouterGemma4` (`RealForwardRunner.swift:1596`), and
   `effectiveScaleBuffers`. Qwen needs a softmax-all → top-8 → renormalize
   router (ceilings already fit: 256 experts, top-8).
3. **Activation.** Every Metal activation site is literal
   `gelu_pytorch_tanh` (`moe.metal:53,368,404`, `prefill.metal:20,636`,
   `utility.metal`); `hiddenActivation` is carried in the manifest but never
   consumed. Qwen needs silu variants.
4. **Attention details.** Scale 1.0 hardcoded (`RealForwardRunner.swift:1546`);
   the K==V full-layer trick is structural (`RealForwardRunner.swift:881,1466`,
   `fused.metal:73-175` bakes a no-scale V norm) — `attentionKEqV` exists as a
   manifest flag but nothing reads it. Qwen: real V (no norm), scale 1/16,
   fused output gate (new kernel), q/k norm on full layers only.
5. **Head + layer tail.** Tied head assumed (`Model.swift:88-94`,
   `RealForwardRunner.swift:1848`); Gemma layer tail (FFN sandwich + layer
   scalar) hardcoded in the fused tail; Qwen tail is
   routed+shared → sigmoid gate → residual. Softcap 30 is a Sampler default
   (`Sampler.swift:104`) not fed from config; embed ×√H hardcoded
   (`RealForwardRunner.swift:911,1449`).
6. **Tokenizer/chat stack — largest non-kernel surface.** Decoder allowlist
   rejects Qwen BPE (`Tokenizer.swift:163-180`), Gemma stop tokens/markers
   hardcoded (`Tokenizer.swift:209-239`, `RawCompletion.swift:356-368`),
   Gemma turn framing (`Tokenizer.swift:336-410`), Gemma tool parser/schema,
   `modelID`/vocab pins (`Tokenizer.swift:46-48,240`).
7. **Arch selection plumbing.** There is no arch identifier in the manifest;
   `ManifestReader.validateArch` (`ManifestReader.swift:160-194`) does
   field-by-field equality against the caller-supplied `ArchConfig`, and every
   entry point defaults to `ArchConfig.gemma4_26B_A4B` (`Model.load` default,
   `Run.swift:194`, `RealInferenceClient.swift:438`, `ServerInference.swift:568`,
   `ConversationIdentity.swift:67`, `AppModelInstallationProbe.swift:22`,
   `DecodeLoadAdmission.swift:22`, `VisionRuntime.swift:171`).

Also: `ContextAdmission` carries two Gemma-measured constants
(non-KV footprint 2 GiB, expert stride 3,358,720 — `ContextAdmission.swift:16-33`),
and the prefill-chunk ring floor uses Gemma vision's 280 pooled tokens
(`RealForwardRunner.swift:279-285`).

## Phases

### Phase 1 — Format + repacker: build `scratch/qwen36.gturbo`

Deliverable: `swift run -c release TurboFieldfareRepack --source qwen36
--output scratch/qwen36.gturbo` produces a verified install; the runtime
still refuses to load it. All Foundation-only, no Metal.

**Status 2026-09-27: items 1-6 implemented (on a machine without a Swift
toolchain — unverified until `swift build` + `Scripts/test.sh` run on macOS).
Item 7 (tests) not yet written.**

1. ✅ **`ArchConfig`**: added defaulted fields `archName` (default
   `"gemma4-26b-a4b"`) and `nonFullLayerKind` (default
   `"sliding_attention"`), and `static let qwen3_6_35B_A3B`
   (`ModelTypes.swift`).
2. ✅ **Manifest format** (`GTurboManifestArchV1`): optional `archName` /
   `nonFullLayerKind` (nil = legacy Gemma manifest, encoded byte-identical);
   `validate` permits `slidingWindow == 0` only for
   `nonFullLayerKind == "linear_attention"` and rejects unknown kinds.
3. ✅ **`ArchInfo.load`**: family branch — configs whose `layer_types`
   contain `linear_attention` (and `model_type` prefix `qwen3_5`) parse the
   Qwen schema (`hidden_act`, `num_experts_per_tok`,
   `shared_expert_intermediate_size`, flat `rope_parameters`, softcap 0,
   sliding window 0); the Gemma branch is unchanged and strict; unknown
   hybrid families fail loudly.
4. ✅ **`RepackPlanner.classify`**: `.mlp.switch_mlp.` routed-expert pattern
   added alongside `.experts.switch_glu.`; Qwen resident slot ordering added
   (ranks 20+, Gemma ranks untouched). `mtp.` tensors are deliberately NOT
   excluded: unknown prefixes hard-error, which is the safe behavior while
   no Qwen build ships MTP weights.
5. ✅ **`RemoteStreamingRepacker.writeManifest` probes**: added `.mlp.gate`
   → router and `.mlp.shared_expert.gate_proj` → sharedExpert. The 8-bit
   `mlp.shared_expert_gate` needs no slot probe; its bits ride its own
   resident-index entry, and parking it in no manifest slot avoids poisoning
   the sharedExpert slot (which drives the runtime's Int4/Int8 kernel
   choice).
6. ✅ **`SupportedModelSource` catalog + `SourceFingerprint`**:
   `gemma4` / `qwen36` entries (qwen36 pinned to revision `38740b8…`,
   index SHA `0b28df60…`), `--source` CLI flag with unknown-id rejection and
   mode guards (vision/discard/verify reject it), Gemma remains the default
   and the vision installer keeps using it.
7. ⬜ **Tests**: synthetic Qwen snapshot (config + `switch_mlp` names,
   40 layers × 256 experts) through the planner; format round-trip with the
   new optional fields; old-manifest backward compatibility. First task of
   the next session, so the Mac run verifies them together with the build.

### Phase 2 — Runtime load + full-attention path

1. Arch selection: manifest `archName` → `ArchConfig` lookup; replace all
   `expecting:` defaults; `validateArch` compares the new fields.
2. `validateQuant` per-arch bits sets (Qwen: attention [4], router [8],
   sharedExpert [4], routedExpert [4], embedding [4]).
3. Per-arch tensor-name mapping in `Model` accessors +
   `validateRuntimeSchema` (Qwen schema: `linear_attn.*` required, no FFN
   sandwich norms, q/k norm only on full layers, `lm_head` required when
   untied, `v_proj` required on full layers).
4. Untied head: `Model.lmHead` accessor; runner head sites use it; schema
   enforces it when `tieWordEmbeddings == false`.
5. Full-attention layer path: V passthrough epilogue variant, attention
   scale from config (1/16), q/k norm, **fused output gate** (split q_proj
   output into gate/head halves, sigmoid), 16Q/2KV head mapping.
6. Silu activation kernels; Qwen router kernels (decode + prefill variants);
   embed scale 1.0; softcap from config (disable at 0).
7. **Parity fixtures**: capture router logits, attention outputs, and MoE
   outputs from an mlx-lm reference run on the Mac; differential tests in the
   `TurboFieldfareValidation` style.

### Phase 3 — Gated delta-rule linear attention

1. State store: per layer, fp32 `S` (32×128×128, ~2 MB) + conv window
   (3×8192 over qkv channels). ~63 MB total at 40 layers → well within
   budget.
2. Decode step kernel: dt = softplus(a_proj(x) + dt_bias), α = exp(-A·dt),
   β = sigmoid(b_proj(x)) (verify exact formulation against transformers
   `qwen3_next` / fla / mlx-lm), rank-1 delta update
   `S ← α·S·(I − β k kᵀ) + β k vᵀ`, output `qᵀS` gated by sigmoid(z), conv,
   out_proj + norm.
3. Chunked-prefill scan (UT/WY transform) — the largest single kernel item;
   port the reference math, verify per-chunk against the decode-step
   recurrence on fixtures, then against mlx-lm.
4. `KVCacheManager`: add the linear-layer kind (state, no KV rows); context
   admission from layout-derived sizes.

### Phase 4 — Prefill perf + integration

1. Chunked GDN prefill through the tile scheduler; MPP for the large
   linear-attn projections (in_proj_qkv is [8192, 2048] — GEMM-shaped).
2. Tokenizer/chat: allow Qwen BPE decoder, `im_start`/`im_end` framing +
   stop tokens from `tokenizer_config.json`, conversation identity, server
   tool schema/parser for Qwen's template, `modelID` plumbing.
3. Cache sizing: extend the allowed slot set (default 16 of 256 = 6.25%;
   whole-pool caching wants ~256 slots/layer → allow up to expertsPerLayer,
   admission math from layout.json expert stride ~1.73 MB).
4. Context admission constants from layout; CLI/app/server arch plumbing;
   benchmarks per COMMUNITY_BENCHMARKS.md.

### Phase 5 — Optional later

MTP self-speculation (needs a source with MTP weights), Qwen vision
companion pack (mirrors the existing pattern; source has the 333 vision
tensors), K2-Horizon-MoVA (see survey; value-expert kernels), and the
bounded streaming quantizer ([idea doc](BOUNDED_STREAMING_QUANTIZER.md)) if
another MLX-less model becomes the target.

## Verification

- Phase 1 is fully testable model-free (planner + format tests) via
  `Scripts/test.sh` on macOS; `TurboFieldfareFormat` is Foundation-only and
  would also build on Linux with a Swift toolchain.
- Phases 2-3 lean on captured-reference differential tests (mlx-lm logits +
  intermediate activations on fixed prompts), then the pinned
  instruction-checkpoint validation and COMMUNITY_BENCHMARKS protocol.
- Expected end state on the 32 GB M1 Pro: whole expert pool cached after
  warmup, decode compute-bound at ~3B active — extrapolated **10-13 tok/s**
  (not a measurement; treat per repo rules).

## Risks / open questions

1. **GDN numerics**: fp32 state and the chunked scan must match the
   reference within tolerance; this is the phase with the highest rework
   risk.
2. **RoPE semantics**: config says `mrope_interleaved` with sections — for
   text-only prompts this should reduce to standard RoPE (partial 0.25);
   confirm against mlx-lm's text path in Phase 2 before writing RoPE code.
3. **Fused q_proj gate order**: which half of the 8192 output is gate vs
   head (first/last 4096) — verify against the reference implementation,
   not assumed.
4. **Tokenizer parity**: Qwen BPE + im_start framing + tool template is a
   wide surface; the server's fail-closed tool parsing must be re-derived,
   not reused.
5. Two couplings to clean up: Gemma vision's 280-token floor in KV ring
   sizing, and the `ContextAdmission` Gemma constants.
