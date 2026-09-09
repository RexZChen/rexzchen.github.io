you type a prompt, press enter, and a paragraph appears.

from the outside, this looks like one event. inside a language model, it is closer to a very fast flipbook: tokens enter, vectors change through dozens of layers, a distribution over the vocabulary appears, one token is chosen, and the whole thing repeats.

so, what could we actually record if we put the model in a very metaphorical MRI scanner?

[[mri-pipeline]]

quick scope note: i am mostly talking about a **decoder-only transformer** at inference time, the family behind many chat LLMs. encoder-decoder models, mixture-of-experts models, and state-space models expose related but not identical machinery.

## first: the weights are mostly the boring part

during ordinary inference, the learned parameters are fixed. they do not rewrite themselves because you asked a clever question. what changes is the **runtime state** produced by applying those parameters to this particular sequence.

let the tokenizer turn the prompt into token IDs

$$
x_{1:n} = (x_1, x_2, \ldots, x_n).
$$

each token becomes an embedding, gets position information, and enters the residual stream. a deliberately simplified layer looks like

$$
\begin{aligned}
a_t^{(\ell)} &= \operatorname{Attn}^{(\ell)}\!\left(\operatorname{LN}(h_t^{(\ell)})\right), \\
m_t^{(\ell)} &= \operatorname{MLP}^{(\ell)}\!\left(\operatorname{LN}(h_t^{(\ell)} + a_t^{(\ell)})\right), \\
h_t^{(\ell+1)} &= h_t^{(\ell)} + a_t^{(\ell)} + m_t^{(\ell)}.
\end{aligned}
$$

real architectures move the normalizations around, add gates, group query heads, use rotary positions, route through experts, and generally refuse to fit neatly in one equation. but the useful picture survives: the model keeps updating a vector for each position by mixing information through attention and transforming it through an MLP.

the phrase **activation** means almost any intermediate tensor produced during that calculation. a **hidden state** usually means the representation at a particular layer and token position. people also say **latent state**, but that is a loose umbrella term, not one canonical tensor hiding behind the API.

## the things we can observe

with access to an open model, or a provider that deliberately returns some internals, we can record quite a lot.

- **input structure:** token IDs, token strings, positions, attention masks, chat-template tokens, and segment labels. this is unglamorous and absolutely essential: two “identical” prompts can become different model inputs.
- **embeddings and residual-stream states:** vectors such as $h_t^{(\ell)}$ for every token and layer. they preserve rich geometry, but an individual coordinate is rarely a clean human concept.
- **attention internals:** queries, keys, values, pre-softmax attention scores, attention weights, and each head's output. for one head,

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + M\right)V,
$$

where the causal mask $M$ blocks future positions.

- **MLP internals:** layer inputs and outputs, pre-activations, nonlinear activations, and, in gated models, the gate values. these are often where feature-oriented or “neuron” analyses begin.
- **normalization statistics:** means, variances, or RMS values used by LayerNorm/RMSNorm. less cinematic, still part of the actual computation.
- **KV cache:** the past keys and values retained at every attention layer so generation does not recompute the whole prefix. a common tensor shape is $[B, H, T, d_{\text{head}}]$ per key/value tensor, though grouped-query attention changes the head dimension.
- **logits:** one unnormalized score $z_v$ for every vocabulary item $v$. the final hidden state is projected through the unembedding matrix:

$$
z = W_U h_n^{(L)} + b.
$$

- **probabilities and uncertainty summaries:** softmax probabilities, entropy, top-token margins, or divergence between two runs. these are derived from logits; they are not additional thoughts the model had.
- **router decisions:** in mixture-of-experts models, the router scores and selected experts are observable too.
- **runtime telemetry:** latency, memory use, cache size, numerical precision, and sometimes kernel-level traces. these tell us how the system executed, not what a sentence “means.”
- **gradients:** normally absent during inference because no backward pass is needed. we can request them in an attribution experiment, but then we are doing extra analysis rather than passively watching a standard generation.

and we usually cannot afford to save all of this. one full activation trace can be enormous. practical tools keep selected hook points, a few token positions, low-dimensional projections, norms, or aggregates.

that compression matters. for example,

$$
\lVert h_t^{(\ell)} \rVert_2
$$

tells us the magnitude of a hidden vector and discards its direction. two states can have the same norm while encoding very different information. a random projection preserves some geometry approximately; it does not preserve a readable copy of the model's “thought.”

## attention is useful, and it is not an explanation certificate

an attention map answers a specific question: how did a head distribute attention mass over allowed source positions for this query position?

that can reveal crisp patterns. heads may focus on delimiters, nearby tokens, repeated strings, or syntactic relations. but high attention weight does not automatically mean “this token caused the answer.” values matter, head outputs are mixed back into the residual stream, later layers can overwrite or route around the result, and very different attention patterns can sometimes produce similar outputs.

so i would read attention as **where one information-routing operation looked**, not as a complete importance score. the classic paper is even titled [Attention is not Explanation](https://aclanthology.org/N19-1357/). the more nuanced conclusion is not “ignore attention”; it is “do not ask one tensor to explain the entire network.”

## the KV cache is memory, but not memory in the human sense

without a cache, autoregressive generation would repeatedly calculate keys and values for tokens it had already processed. because causal attention prevents an earlier position from depending on later ones, those past keys and values can be reused.

at decode step $t$, conceptually we append the new entries:

$$
K_{1:t} = [K_{1:t-1}; K_t],
\qquad
V_{1:t} = [V_{1:t-1}; V_t].
$$

the next query $Q_t$ attends over that retained prefix. the cache therefore grows with sequence length in ordinary full attention, consumes a lot of memory, and makes each next-token step much faster.

but a large KV norm is not “more memory,” better recall, or deeper reasoning. it is a property of stored vectors. and the KV cache is temporary runtime state, not the model's long-term knowledge: delete the request state and it is gone. learned knowledge lives primarily in the weights; retrieved documents and conversation history arrive as more input tokens.

## logits are the last observable before a choice

the logits are often the cleanest place to watch the answer taking shape. convert them into a probability distribution with temperature $\tau$:

$$
p(v \mid x_{\le t})
= \frac{\exp(z_v/\tau)}{\sum_u \exp(z_u/\tau)}.
$$

lower temperature sharpens the distribution; higher temperature flattens it. entropy summarizes concentration:

$$
H(p) = -\sum_v p(v)\log p(v).
$$

lower entropy means the distribution is more concentrated. it does **not** mean the answer is true, understood, or correct. a model can be confidently wrong with beautiful numerical precision.

we can also project intermediate hidden states through the final unembedding matrix. this is usually called a **logit lens**. it asks which tokens an intermediate representation resembles under that projection. useful probe, derived view, not the literal output distribution produced at that layer.

## okay, so what exactly is decoding?

people use “decode” in two related ways:

- a tokenizer decodes token IDs back into text
- a generation algorithm decodes model scores into a token sequence

the second meaning is the important one here. the neural network produces logits; the **decoding policy** decides what to do with them.

the usual loop is:

- run **prefill** on the prompt, processing its positions in parallel under a causal mask and filling the cache
- take the logits at the final prompt position
- apply any score processors: temperature, repetition penalties, banned tokens, grammar constraints, top-$k$, top-$p$, and so on
- choose a token
- append it to the sequence
- run one-token **decode**, reusing the KV cache
- stop on an end token, a stop string, or a length limit; otherwise repeat

greedy decoding always chooses

$$
x_{t+1} = \arg\max_v p(v \mid x_{\le t}).
$$

sampling draws $x_{t+1}$ from a distribution, often after top-$k$ or nucleus filtering. beam search keeps several sequence hypotheses. same model, same prompt, different decoder: potentially very different text.

this is why “the model said X” is incomplete experimental reporting. model revision, chat template, prompt, temperature, filters, seed, maximum length, and stopping rules all belong in the record.

## what happens without decoding?

this phrase can mean three different things, which is mildly annoying.

**no generation:** we can run a forward pass on a supplied sequence and stop at its logits or hidden states. this is how we inspect a prompt, create embeddings, classify with a head, or score known text. no new token has to be chosen.

**teacher-forced scoring:** if the continuation is already known, all of its token probabilities can be evaluated in parallel under a causal mask. we observe how likely the model found each supplied next token, but the model does not decide the sequence.

**generation without a KV cache:** decoding still happens. the implementation simply recomputes the entire prefix on every step instead of reusing past keys and values. barring small numerical differences from kernels or precision, the mathematical next-token distribution should be the same; the work is just much more expensive.

there is no useful version of open-ended generation with literally no selection rule. even “just take the output” usually means greedy decoding was selected by default.

## “thinking” versus “not thinking” is not computation versus no computation

a direct-answer model still runs every input through its layers. an explicit-reasoning mode additionally generates intermediate tokens, and those tokens become context for later tokens.

so visible chain-of-thought is generated text, not an exhaustive dump of hidden activations. turning it off does not turn internal computation off. turning it on can change the prompt template, the prefix, the number of decode steps, and every later state.

when comparing the two, use the same checkpoint and sampling settings where possible, save the rendered chat templates, and be honest about alignment. “step 12” in two diverged generations is not automatically the same semantic moment.

## here is the MRI, metaphorically

the widget below compresses sequence positions, layers, and measured components into a replayable view. try the hidden-state, attention, memory/KV, MLP, and output-probability tabs; then compare direct and explicit-reasoning runs.

[[mri-widget]]

the default traces are illustrative and say so in the interface. that is deliberate. fake precision is worse than a useful cartoon wearing a name tag. the widget can also load a compatible recorded trace, where the displayed values come from captured tensors and summaries.

## observation is not causation

suppose one layer's activation norm spikes just before the model emits the correct city. we have observed a correlation. we have not shown that the spike represents the city or caused the answer.

for stronger evidence, intervene. replace an activation from a clean run with the corresponding activation from a corrupted run, ablate a head, patch a residual-stream vector, or change the input while holding everything else fixed. then measure the output change.

even intervention requires care: a model may have redundant paths, an ablation can push it off-distribution, and results depend on the metric and patch site. but this moves us from “the tensor looked interesting” toward “changing this state changed the behavior.”

the useful ladder is:

- raw tensors tell us **what values occurred**
- summaries tell us **where something changed**
- probes tell us **what information is decodable**
- controlled interventions test **what made a difference**
- a replicated circuit-level account tries to explain **how the behavior is implemented**

those are different claims. a good visualization should make the gaps harder to forget.

## my capture checklist

if i were recording an inference trace for an actual experiment, i would save:

- exact model ID and immutable revision
- tokenizer and fully rendered chat template
- raw prompt token IDs and decoded token strings
- dtype, quantization, device, and library versions
- generation parameters, random seed, and stopping reason
- selected hidden, attention, MLP, router, and cache measurements with tensor shapes
- raw final logits or a clearly labeled top-candidate-plus-other approximation
- prefill and generation timings
- the projection or aggregation recipe, including its seed
- provenance on every view: recorded, derived, projected, or illustrative

and i would avoid naming the brightest cube “understanding.” that word is carrying enough already.

## places to keep digging

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), a friendly visual introduction
- [Hugging Face: caching](https://huggingface.co/docs/transformers/cache_explanation), including KV shapes and custom generation loops
- [Hugging Face: generation strategies](https://huggingface.co/docs/transformers/generation_strategies), for greedy, sampling, and beam decoding
- [TransformerLens main demo](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html), for caching and intervening on activations
- [What Does BERT Look At?](https://aclanthology.org/W19-4828/), a concrete analysis of attention patterns
- [Attention is not Explanation](https://aclanthology.org/N19-1357/), the necessary warning label
- [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042), for causal-intervention details and footguns

the short version: an LLM does not contain one glowing “thought state.” it produces many tensors, at many layers and positions, while a decoding policy repeatedly turns the final scores into new context.

we can watch a surprising amount of that process. the rigor is in remembering exactly what each measurement can, and cannot, tell us.
