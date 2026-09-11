your phone can run a small language model quickly, or a big one slowly.

the obvious move is to pick one: fast and a bit dim, or smart and painfully slow. speculative decoding refuses to choose. the small model guesses a few tokens ahead, the big model checks the whole guess in one pass, and what comes out is the big model's text. not roughly. exactly, in distribution.

that sounds like it should break some conservation law. it does not, and the reason is a nice mix of hardware and probability. it also gets a lot more interesting once the two models stop living on the same chip, which is the normal situation at the edge.

[[spec-round]]

same setting as the MRI note below: a decoder-only transformer, generating one token at a time.

## why decoding is slow in the first place

generation is a loop. one forward pass, one token, append, repeat. for a dense model, every step reads essentially all of the weights, and at batch size one, which is exactly what a phone serving one person looks like, the expensive part of that step is not the arithmetic. it is moving the weights from memory to where the arithmetic happens. [Pope et al.](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html) put it plainly: at small batch sizes, the time to load weights dominates.

a crude but honest bound:

$$
\text{tokens per second} \;\lesssim\; \frac{\text{memory bandwidth (GB/s)}}{\text{weights read per step (GB)}}.
$$

Apple quotes 120 GB/s of memory bandwidth for its base M4 chip. an A100 80GB is rated at about 2,000 GB/s. so a 4 GB model tops out near 30 tokens per second on the laptop chip and past 500 on the data-center one, before any compute is counted. on the small chip, faster math kernels barely help, because it spends much of each step waiting for bytes.

that waiting hides a useful asymmetry. while the weights stream in, the compute units are mostly idle, so asking them to handle five positions instead of one adds surprisingly little time. it is the same parallelism that lets prefill push a whole prompt through the layers at once, or lets teacher-forced scoring evaluate a known continuation in a single pass.

so writing five tokens takes five slow passes. *checking* five proposed tokens takes about one.

## guess, then check

call the big model the **target**, with next-token distribution $p$, and the small model the **draft**, with distribution $q$. one round goes like this:

- the draft proposes $\gamma$ tokens, one cheap autoregressive step at a time
- the target runs a single forward pass over the prefix plus all $\gamma$ guesses, which gives its own distribution at $\gamma + 1$ positions
- walk the guesses left to right, keep each one that passes the acceptance test, and stop at the first failure
- at the stopping point the target contributes one token itself: a correction after a rejection, or a free bonus token if every guess survived
- roll the caches back past the rejection and start the next round

every round ends with at least one token that came straight from the target, so the worst case is ordinary decoding plus some wasted guessing. the best case is $\gamma + 1$ tokens for one target pass and a little drafting.

with greedy decoding the acceptance test is simple: keep a guess if it matches the target's argmax at that position. the output is the same token sequence the target would have produced alone.

with sampling, the test is more interesting.

## why the output is still the big model's

accept a guessed token $x$, which was drawn from $q$, with probability

$$
\min\!\left(1,\ \frac{p(x)}{q(x)}\right).
$$

if it is rejected, draw a replacement from the leftover distribution

$$
r(x) = \frac{\max\big(0,\ p(x) - q(x)\big)}{1 - \alpha},
\qquad
\alpha = \sum_x \min\big(p(x),\ q(x)\big).
$$

a rejection happens with probability $1 - \alpha$. now add up both ways token $x$ can come out:

$$
\underbrace{q(x)\,\min\!\left(1, \tfrac{p(x)}{q(x)}\right)}_{\text{guessed and kept}}
\;+\;
\underbrace{(1 - \alpha)\, r(x)}_{\text{rejected, then redrawn}}
\;=\;
\min\big(p(x), q(x)\big) + \max\big(0,\ p(x) - q(x)\big)
\;=\; p(x).
$$

that is the entire trick. wherever the draft over-guesses a token, the excess gets rejected. wherever it under-guesses, the leftover distribution tops it back up. the draft decides *how fast* you move, never *where you end up*.

[Leviathan et al.](https://arxiv.org/abs/2211.17192) at Google and [Chen et al.](https://arxiv.org/abs/2302.01318) at DeepMind arrived at this rule independently, building on the greedy guess-and-check of [blockwise parallel decoding](https://arxiv.org/abs/1811.03115). the guarantee is also why it could ship quietly: [Google says](https://research.google/blog/looking-back-at-speculative-decoding/) it runs behind AI Overviews in Search, where nobody wants a faster but slightly different answer.

[[spec-mass]]

$\alpha$ doubles as the expected acceptance rate, and it equals one minus the total variation distance between $p$ and $q$. a draft that thinks like the target gets accepted a lot. a draft with different habits wastes work, but it cannot change the answer.

## how much faster, roughly

pretend each guess is accepted independently with probability $\alpha$. wrong in detail, useful in spirit. the expected number of tokens per target pass is then

$$
\tau(\gamma) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}.
$$

let $c$ be the cost of one draft step relative to one target step. a round costs $\gamma c + 1$ target steps of time, so the expected speedup is

$$
S = \frac{1 - \alpha^{\gamma + 1}}{(1 - \alpha)(\gamma c + 1)},
$$

which is the walltime formula from the Leviathan et al. paper. three things fall out of it:

- **acceptance is the big lever.** with $\alpha = 0.8$, $\gamma = 4$, and $c = 0.1$, a round yields $\tau \approx 3.4$ tokens and $S \approx 2.4\times$. drop to $\alpha = 0.5$ and the same setup gives about $1.4\times$.
- **draft length has a sweet spot.** the chance that a whole draft survives decays like $\alpha^\gamma$, while the cost of drafting keeps growing linearly.
- **draft cost punishes big drafts.** a draft half the size of the target is a very precise way to waste time.

acceptance depends on the text as much as on the models. boilerplate, code, structured output, and anything that copies from the prompt are easy to guess. in the original experiments it was usually lower with sampling than with greedy decoding: on one translation task, 0.75 at temperature zero against 0.62 at temperature one, and a 3.4× speedup against 2.6×.

the race below puts both models on the same phone. same sentence, two strategies.

[[spec-race]]

run it a few times. the measured speedup wobbles because acceptance is random, and it lands a little under the prediction on average, because a 32-token sentence runs out in the middle of a round. the final sentence never changes.

## what it saves, and what it quietly costs

the savings are real:

- **big-model passes.** the target runs $1/\tau$ as often, and at batch size one that is most of the latency.
- **energy, usually.** fewer long, memory-bound passes tend to mean less energy per token. but verification works the chip harder while it lasts, so the saving shrinks at low acceptance and can disappear.
- **server capacity, when the target is remote.** one verification pass per round instead of one pass per token.

so are the costs:

- **memory.** the draft needs its own weights and its own KV cache. in a data center that is a rounding error. on a phone with 8 to 16 GB shared with the operating system, the camera, and forty browser tabs, it is the whole negotiation.
- **wasted work.** rejected guesses were computed for nothing, and so were the target's positions after the rejection.
- **verification is only nearly free.** "five positions cost about one" assumes idle compute, and small chips have much less of it. Apple's [ReDrafter](https://arxiv.org/abs/2403.09919) is a nice data point: up to 2.3× on Apple silicon, but the best number of candidate drafts to check per pass was 1 to 3 there, against 50 or more on server GPUs.
- **plumbing.** caches roll back, both models need the same tokenizer, and batching changes the math. once a busy server is compute-bound, extra positions stop being free, and under [heavy load or poor acceptance](https://arxiv.org/abs/2406.14066) speculation can even add latency.

that last point is worth underlining for edge people. speculative decoding is at its best exactly where a personal device lives: one user, one stream, memory-bound.

## where do the models live?

in a data center, draft and target sit next to each other. at the edge you get to choose, and every choice changes a different term of the same equation. per-token latency is the cost of one round divided by the tokens it produces:

$$
L \;=\; \frac{\gamma\, t_{\text{draft}} \;+\; t_{\text{verify}}(\gamma) \;+\; t_{\text{net}}}{\tau(\gamma)}.
$$

- **phone only, speculative.** both models on the device, $t_{\text{net}} = 0$. works offline, private by construction. the target has to fit, which usually means a modest quantized model, and the draft has to fit next to it. [LLMCad](https://arxiv.org/abs/2309.04255) pushes this further: a small model in memory drafts a token tree, and a model too large for memory verifies it.
- **phone drafts, edge server verifies.** the phone runs only the small model and a nearby server runs a much bigger target. you sample from the big model's distribution with a small model's memory footprint, but every round now pays a network round trip.
- **cloud only.** the server generates and streams. one round trip up front, then each token costs a server step. best models, zero local memory, and nothing works in a tunnel.
- **phone only, draft model.** no verification at all. fastest, cheapest, and you get exactly the small model's quality.

the split setup has a neat and slightly cruel property. because every round costs a round trip, the best draft length *grows* with latency: when trips are expensive, you want more tokens per trip. but long drafts rarely survive intact, so past some round trip, plain cloud streaming simply wins on speed.

it still has two cards to play. the server does one pass per round instead of one per token while the phone's idle silicon does the guessing, which matters a lot to an operator with a handful of edge GPUs and many devices. [SpecEdge](https://arxiv.org/abs/2505.17052) reports 2.22× server throughput when consumer-grade edge GPUs do the drafting. and when the network dies, the phone can keep talking with its draft alone. worse, but alive.

one detail that is easy to miss: exact *sampling* across the split needs more than token IDs. to redraw from the leftover distribution, the server needs the draft's probabilities, in general a whole vocabulary's worth for every guessed position. greedy verification only needs the IDs. real systems [truncate what they send](https://arxiv.org/abs/2505.11788), and each shortcut there is a small step away from exactness.

## so where is the speed-quality trade-off?

this is the part i find clarifying.

**plain speculative decoding does not trade quality for speed. it trades memory, wasted compute, plumbing, and sometimes round trips.**

quality is pinned to whichever target does the verifying. the trade-off comes in through three other doors, and at the edge you usually walk through at least one of them:

- **which target can you afford to verify with?** on the phone, a modest quantized model. on the edge server, something much bigger. speculation makes either one faster. it does not make the phone's target as good as the server's.
- **do you always check?** lossy variants keep tokens the target would have rejected. [Medusa](https://proceedings.mlr.press/v235/cai24b.html) offers a typical-acceptance rule that keeps plausible-enough candidates, [Big Little Decoder](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html) hands control to the big model only when the small one looks unsure, and [U-HLM](https://arxiv.org/abs/2412.12687) skips both the uplink and the big model for tokens the phone is confident about, cutting transmissions by about 46% while keeping 97.5% of the big model's accuracy. every skipped check buys speed, and across a network a skipped round trip, with tokens that never faced the veto.
- **what happens without a network?** a split system that falls back to its draft stays available, at the draft's quality.

the lab below puts those doors on one map. the hardware profile is illustrative, not a benchmark. move the round trip, the acceptance rate, and the leniency, then cut the network.

[[edge-lab]]

things worth trying:

- set the round trip to 10 ms, then 80 ms, and watch 04 and 05 trade places
- drop acceptance to 0.35: speculation on the phone barely beats the big model alone, because most guesses get thrown away
- push acceptance to 0.9 and watch the split setup tolerate a far longer round trip, with a longer best draft
- raise leniency: the speculative setups get faster and pick up a growing share of unvetted tokens
- cut the network: 04 falls back to the draft row, and 05 has nowhere to go

## the drafter zoo, sorted by memory

since memory is the scarcest thing at the edge, i like sorting draft strategies by what they add to RAM.

- **zero parameters: copy from context.** [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) proposes continuations by matching recent tokens against the prompt, which is shockingly effective for summaries, retrieval-augmented answers, and code edits, where the output copies spans of the input. [lookahead decoding](https://proceedings.mlr.press/v235/fu24a.html) builds n-gram guesses on the fly from the target's own parallel iterations.
- **zero extra weights: the target's own shallow path.** [Draft & Verify](https://aclanthology.org/2024.acl-long.607/) drafts by skipping intermediate layers of the target. [LayerSkip](https://aclanthology.org/2024.acl-long.681/) trains the model so that early exits make good drafts and share work with verification.
- **a few extra weights: heads.** [Medusa](https://proceedings.mlr.press/v235/cai24b.html) bolts several decoding heads onto the target. [EAGLE](https://proceedings.mlr.press/v235/li24bt.html) drafts at the feature level with one lightweight layer and gets unusually high acceptance.
- **a whole second model.** the classic recipe: most flexible, needs a shared tokenizer, costs the most memory. [DistillSpec](https://openreview.net/forum?id=rsY6J3ZaTF) and [online speculative decoding](https://proceedings.mlr.press/v235/liu24y.html) raise acceptance by teaching the draft to imitate the target, which on a device could mean imitating it on one person's traffic.

orthogonal to all of these are **token trees**: guess several branches and verify all of them in one pass with a tree-shaped attention mask ([SpecInfer](https://dl.acm.org/doi/10.1145/3620666.3651335), [EAGLE-2](https://aclanthology.org/2024.emnlp-main.422/), [Sequoia](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html)). more tokens per pass, more work per verification. on a chip with little headroom, trees should stay small. but when the target is offloaded to slow memory and one pass takes seconds, trees can get enormous: [SpecExec](https://arxiv.org/abs/2406.02532) accepts up to 20 tokens per target pass and runs a 4-bit 70B model at over 5 tokens per second on a single RTX 4090.

## "exact" has fine print

- **exact in distribution, not identical across runs.** with sampling, tokens follow the target's distribution. two runs still differ, as they would without speculation.
- **greedy is token-identical in math, nearly identical in practice.** verification runs $\gamma + 1$ positions through kernels with different shapes than a one-token step, and floating point can flip a near-tie. [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) has a good write-up of why even temperature zero is not deterministic in practice.
- **lossy variants are a different contract.** relaxing acceptance is a legitimate speed-quality trade. just call it one.
- **the target is whatever you verified with.** a 4-bit on-device target has 4-bit quality. speculation makes it faster, not smarter.

## my edge checklist

if i were shipping this on a device, or across a device and an edge server, i would measure:

- acceptance on real traffic, split by task and temperature, not on a benchmark
- the actual cost of verifying $\gamma + 1$ positions on the actual chip
- the full memory bill: both sets of weights and both KV caches at the longest supported context
- energy per token and thermal behavior over minutes, since a throttled chip quietly changes $c$
- the round-trip distribution, not its mean, because the tail decides whether a split round feels smooth
- radio energy, since a cellular radio lingers in a high-power state after every transmission ([about 11.5 s on LTE](https://dl.acm.org/doi/10.1145/2307636.2307658)), and rounds that arrive faster than that never let it sleep
- the fallback policy when the network degrades, and whether users can tell
- whether any acceptance rule is lossy, and a quality metric that would catch the drift

## the short version

decoding is slow because every token needs a full pass through the big model, and at batch size one that pass mostly waits on memory. checking many tokens costs about as much as writing one. speculative decoding exploits that asymmetry: a cheap model guesses, the expensive model verifies in parallel, and a small rejection rule guarantees the result is distributed exactly as if the expensive model had worked alone.

at the edge, the question shifts from "is it faster?" to "which term of the round am i paying for?" drafting, verification headroom, round trips, or memory. and the speed-quality trade-off does not live inside the algorithm. it lives in which target you can afford to verify with, and in how often you decide not to check.

## sources worth opening

- [Stern et al., Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115), NeurIPS 2018
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023
- [Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023
- [Google Research, Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/), 2024
- [Pope et al., Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html), MLSys 2023
- [Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://aclanthology.org/2024.findings-acl.456/), Findings of ACL 2024
- [Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://proceedings.mlr.press/v235/cai24b.html), ICML 2024
- [Li et al., EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://proceedings.mlr.press/v235/li24bt.html), ICML 2024
- [Cheng et al., Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/abs/2403.09919), 2024
- [Xu et al., LLMCad: Fast and Scalable On-device Large Language Model Inference](https://arxiv.org/abs/2309.04255), 2023, later EdgeLLM in IEEE TMC 2025
- [Svirschevski et al., SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices](https://arxiv.org/abs/2406.02532), NeurIPS 2024
- [Oh et al., Uncertainty-Aware Hybrid Inference with On-Device Small and Remote Large Language Models](https://arxiv.org/abs/2412.12687), ICMLCN 2025
- [Park et al., SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs](https://arxiv.org/abs/2505.17052), NeurIPS 2025
- [He, Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), Thinking Machines Lab 2025
