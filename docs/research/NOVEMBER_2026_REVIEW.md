# November 2026 review: new models and Qwen3.6 engine speed

Date: 2026-11-05. Machine: Apple M1 Pro, 32 GB, macOS 26.6.2 (25G83),
Swift 6.4.2. Base commit: local `main` at `0d715ee` (Qwen3.6 work, not on
`origin`), plus the changes listed at the end, committed with this note.

The machine was in normal use during every run: a virtual machine, browsers,
editors, about 15 GB free memory and 7 of 8 GB swap in use. Treat all numbers
as measurements under that load, not best cases. Every A/B below alternated
the two sides in the same session.

## 1. Upstream changes

`origin/main` has four releases we have not merged (0.9.2 to 0.9.4):

| Commit | Change |
| --- | --- |
| `77e8f9c` (#186) | SwiftUI macro fix for Command Line Tools builds |
| `4d54604` (#191) | Selectable model locations and a self-built Mac app bundle |
| `2e9ea50` (#192) | Decode-service teardown write protection (0.9.3) |
| `d994a65` (#195) | Code-block copying and chat fixes (0.9.4) |

That is 88 files, +4590/-457 lines. A trial merge (`git merge-tree`) conflicts
in four files, because upstream's model-location work and our multi-model work
touch the same places:

- `Sources/TurboFieldfareApp/Core/Installation/AppModelLocation.swift`
- `Sources/TurboFieldfareApp/Core/State/AppModel.swift`
- `Sources/TurboFieldfareApp/Mac/Installation/ModelInstallView.swift`
- `Sources/TurboFieldfareServer/Core/ServerPromptCache.swift`

Not merged: resolving it means choosing between two designs for picking a
model, which is the owner's call. None of the four upstream changes touch the
decode engine.

## 2. New models

Goal: at least as capable as Qwen3.6-35B-A3B and at least as fast on this
machine. Speed here is mostly set by how many expert bytes each token needs
and how much of the expert pool fits in RAM. Qwen3.6 needs about 0.57 GB of
experts per token from an 18.1 GB pool.

Result: **nothing new beats Qwen3.6 on both counts.** Details:

| Model | Released | Size (total / active) | Verdict |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | (current) | 35B / 3B | Still the newest small Qwen MoE. No Qwen3.7 or Qwen4 weights; Qwen3.8-27B is dense, so every token reads all 27B. |
| Kolibri-1 (Aleph Alpha) | 2026-10-03 | 78.1B / 3.46B | Vendor numbers beat Qwen3.6 on general, math and code (75.5 vs 71.4, 96.5 vs 87.8, 89.3 vs 87.7) but trail on SWE-bench Verified (66.4 vs 73.8), Terminal-Bench and long context. The 4-bit expert pool is about 42-44 GB, so on 32 GB most experts would come from disk every token. Expect it to be slower than Qwen3.6, not faster. It also needs new engine work: sliding-window KV cache, layers without RoPE, a sigmoid router without renormalization. MLX 4-bit and 3-bit builds exist if a larger-RAM machine is ever the target. |
| Nemotron 3.5 Bolt 74B-A9B (NVIDIA) | 2026-10-15 | 74B / 9B | Strong scores, but 3x the active weights of Qwen3.6, a Mamba-2 hybrid (new kernels), and no MLX build. Slower. |
| K2-Horizon-MoVA-36B-A4B | (Sept survey) | 36B / 4B | Better on agentic benchmarks, but 4B active plus routed value experts means more bytes per token, and there is no MLX 4-bit source. Not faster. |
| Ornith-1.5-35B-A3B | ? | 35B / 3B | Mentioned online; no config or weights could be verified. |

Everything larger (GLM, MiMo, Mistral Small 4, Nemotron Super/Ultra, Qwen3.8
Flash) and Qwen3.6 fine-tunes were ruled out as in the September survey.

## 3. Where Qwen3.6 decode time goes

I added opt-in counters (`TURBO_EXPERT_STATS=1`, printed by the CLI after the
footer). Short prompt, 256 tokens, default 16 expert slots per layer:

- About 140 ms per token in total.
- About 59-68 ms is GPU work. The GDN (linear-attention) layers are the
  largest part: about 1.07 ms each for 30 layers, versus 0.95 ms for each full
  attention layer, 0.46 ms for routed experts and 0.19 ms for the shared
  expert, per layer.
- About 34-36 ms is reading missing experts from disk or the OS file cache.
  The GPU sits idle while this happens. Only 47% of expert requests hit the
  16-slot cache.
- About 7 ms is the final "pick the next token" step (norm plus the
  248K-word output matrix).

Two probes, in `Tests/TurboFieldfare/Core/Kernels/Quant/Int4GEMVThroughputProbe.swift`
(skipped unless `TURBO_PERF_PROBE=1`):

- The int4 matrix-vector kernel alone reaches 95 GB/s at 8192x2048,
  85 GB/s at 4096x2048, 56 GB/s at 2048x4096 and 25 GB/s at 512x2048. Inside
  the model the GDN layer is about twice as slow as the sum of its parts
  would suggest.
- Each kernel call opens its own compute encoder. A dispatch costs 11.1 us
  with its own encoder versus 4.2 us when encoders are shared, so the roughly
  800 encoders per token cost about 5 ms per token.

## 4. Experiments

| # | Change | Result | Kept? |
| --- | --- | --- | --- |
| E1 | More expert slots (32, 64 per layer) | 16: 7.73 / 7.06 tok/s; 32: 7.37 / 7.54; 64: 5.95 / 6.99. No gain; more slot memory leaves less room for the OS file cache. | No |
| E2 | Shape-specialized GPU kernels for Qwen's sizes (the existing ones only cover Gemma) | 7.16 tok/s vs 7.07-7.73. Within noise. | No (reverted) |
| E3 | **Next-layer expert prefetch** (below) | 256 tokens: 7.07 / 7.73 off vs 9.08 / 9.10 on. Same text. | **Yes, on by default** |
| E4 | E3, but start the background read before the current layer's own read | 8.79 / 9.18 tok/s. The two reads compete, and the current one is the one the GPU is waiting for. | No |
| E5 | E3 with 32 slots | 8.49 / 8.43 tok/s, slower than 16 slots. | No |
| E6 | E3, plus encoding layer L+1's attention commands while layer L's experts load | 9.22 / 10.24 tok/s vs 9.69 / 10.09 without. Same text. The waiting just moved into the next GPU wait. | No |
| E7 | E3, plus queueing the shared-expert work before the CPU waits on the router | Three rounds: 10.07 / 9.17 / 8.81 vs 9.71 / 8.98 / 9.03 (+1%, noise). Same text. | No |

### E3: next-layer expert prefetch

While the GPU runs layer L's attention, it also runs layer L+1's router on
layer L's post-attention state. That is one extra RMS norm and one small
256x2048 matrix-vector product. When the CPU reads layer L's chosen experts, it
also reads the guess for L+1 and starts loading any of those experts that are
not cached on a background thread. By the time layer L+1 asks for its
experts, most are already in memory.

- 81% of guesses are correct (64,366 of 79,560 expert picks).
- Time blocked on expert reads drops from about 9.3 s to 4.2 s per 256 tokens.
- The output does not change: the prefetch only warms the cache, it never
  changes what is computed. A greedy 62-token story was byte-identical with
  the feature on and off (8.38 -> 10.28 tok/s).
- Memory cost: three tiny buffers. No new expert memory; prefetched experts
  go into the existing 16 slots.
- Turn it off with `TURBO_QWEN_PREFETCH=0`.

Full benchmark protocol (1024 tokens, the three standard prompts, two
alternating rounds):

| Prompt (prompt / new tokens) | Off, tok/s (r1, r2) | On, tok/s (r1, r2) | Mean change |
| --- | --- | --- | --- |
| short-explanation (60 / 1024) | 8.395, 8.244 | 9.799, 10.185 | +20% |
| medium-review (424 / 1024) | 7.993, 8.025 | 10.000, 9.695 | +23% |
| long-synthesis (2938 / 1024) | 6.960, 6.704 | 8.810, 8.794 | +29% |

Command for each run (prefetch off adds `TURBO_QWEN_PREFETCH=0`):

```bash
.build/release/TurboFieldfareCLI --model scratch/qwen36.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/<case>.json \
  --max-new 1024 --max-context 4096 --temperature 0.2 --top-k 64 --top-p 0.95 \
  --seed <20260721|20260722|20260723>
```

All twelve runs exited 0.

### Where the time goes with E3 on

One 256-token run at 10.3 tok/s (about 97 ms per token):

- 58 ms GPU work.
- 14 ms blocked on the experts the guess missed.
- 6 ms on the final norm, output matrix and sampling.
- 1 ms waiting for background reads to finish. The prefetch keeps up.
- About 16 ms where the GPU is idle between the three command buffers of each
  layer. E6 and E7 show this is not CPU encoding time. The best explanation
  is the per-layer CPU-GPU round trip: the CPU must see the router's choice
  before it can load and bind experts. Removing that round trip needs the
  GPU to pick experts on its own, which needs the whole 18 GB expert pool in
  memory. That does not fit next to normal desktop use on 32 GB.

## 5. Ideas not tried yet, by expected payoff

1. **One shared compute encoder per command buffer.** About 5 ms per token
   (around 5%) from the probe numbers. Needs every kernel's `encode` to accept
   an encoder instead of a command buffer, so it is a wide but mechanical
   change.
2. **Fuse the small GDN kernels.** GDN layers are about half the GPU time and
   run at about half the speed their individual kernels manage in isolation.
3. **Raise prefetch recall.** Guessing the top 10-12 experts instead of the
   top 8 should catch some of the 19% misses, at the cost of extra reads.
   Layer 0 of the next token is never predicted yet.
4. **MTP self-speculation** (`docs/research/QWEN36_MTP_SELF_SPECULATION.md`).
   Biggest possible win, biggest change. Run that document's gating
   measurement first.

## 6. Changes committed with this note

- `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift`:
  next-layer prediction in Qwen decode (on by default), plus opt-in timing.
- `Sources/TurboFieldfare/Runtime/Inference/ModelExpertIO.swift`:
  `prefetchRoutedExperts(layer:experts:group:)`.
- `Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift`:
  `ExpertIOStats`, opt-in counters (`TURBO_EXPERT_STATS=1`).
- `Sources/TurboFieldfareCLI/Run.swift`: prints those counters after the
  footer when enabled.
- `Tests/TurboFieldfare/Core/Kernels/Quant/Int4GEMVThroughputProbe.swift`:
  the two probes, skipped unless `TURBO_PERF_PROBE=1`.

Protocol deviations: every 1024-token run stopped at `maxTokens`, not
`endOfTurn`. Qwen3.6 always thinks and the thinking is hidden, so the visible
answer was still unfinished at 1024 tokens; the speed numbers are valid but
the answer text cannot be checked. No separate warm-up runs were discarded;
the alternating order spreads cache warmth across both sides. The machine was
under normal desktop load.
