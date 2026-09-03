okay, imagine this.

you give a model one oddly specific sentence from a paper, and it completes the next line almost perfectly.

your first reaction is probably: *hey, you have seen that, haven't you?*

fair. but scientifically, that sentence hides at least three different questions. and unfortunately, a confident completion does not answer any of them by itself.

[[seen-it-triad]]

## three things we keep collapsing into one

let the model be M, the candidate document be D, and the current query be q.

**exposure** asks whether D, a duplicate of D, or a close variant occurred somewhere in training. this is a question about the data-generating history of M.

**memorization** asks whether training caused M to retain information unusually specific to D. a document can be present without leaving a detectable trace. the same fact can also be reinforced by thousands of other documents, even if this exact one was absent.

**causal use** asks whether a D-specific trace actually contributes to M's answer for q. a model can store something and still solve the present problem using a more general computation.

so the clean version is:

**exposure != memorization != causal use at inference**

these variables are related, but none is a synonym for another. "seen" is not one bit sitting somewhere in the weights waiting for us to read it.

## the first temptation: ask how surprised the model is

for an autoregressive language model, a sequence x = (x1, ..., xn) receives likelihood

**log pM(x) = sum over i of log pM(xi | x before i).**

if training raised the probability of x, perhaps a member should look less surprising than a non-member. that intuition is useful, but raw likelihood is badly confounded. plain prose, repeated facts, familiar genres, short passages, and easy tokenization can all make unseen text look familiar.

this is why [Min-K% Prob](https://arxiv.org/abs/2310.16789) does not simply average every token. it focuses on the least likely tokens, where rare names, numbers, or unusual phrases may carry a sharper membership signal. [Min-K%++](https://arxiv.org/abs/2404.02936) later proposed a theoretically motivated local-maximum view of the same detection problem.

try the toy probe below. move k and watch which tokens actually determine the score.

[[min-k-lab]]

the useful intuition is that common tokens mostly tell us that the model knows the language. the surprising tokens are more diagnostic. still, a score is evidence under a particular calibration, not a certificate of membership.

there is another wrinkle: token position matters. in an autoregressive model, later words become easier once enough context has accumulated. [PDR](https://aclanthology.org/2026.acl-long.562/) reports that membership signal is often concentrated in early, high-entropy positions and introduces positional decay reweighting to emphasize that evidence.

in compact form, many detectors are variations of

**score(x) = sum over i of wi times si,**

where si is token-level evidence and wi decides which tokens or positions matter most.

## one paragraph is a hostile little statistical problem

suppose a detector says your paragraph was probably in training. what else could explain the score?

- the paragraph is simply easy to predict
- its style matches the model's training distribution
- the same fact appears in many other places
- a near-duplicate exists, but not this exact document
- member and non-member samples come from different time periods or domains
- the model was fine-tuned on related material after pretraining

that last list is not theoretical fussiness. a large evaluation of membership inference attacks by [Duan et al.](https://arxiv.org/abs/2402.07841) found that many attacks barely beat random guessing across their tested LLM settings, and that apparent success can come from distribution shift between members and non-members.

so if the training corpus is hidden, the honest output is usually not "yes, seen." it is closer to:

**this sample is more member-like than matched controls under detector S, with uncertainty U.**

not as tweetable, admittedly. much more defensible.

dataset-level evidence is often stronger because we can aggregate many related examples, estimate error, and compare against controls. recent work such as [FTD](https://aclanthology.org/2026.acl-long.1390/) frames contamination filtering with explicit false discovery rate control. that is the direction i trust more: predeclare an error budget, then make a statistical decision.

## perturb the surface, preserve the problem

likelihood asks whether the exact string looks special. generalization tests ask whether the behavior survives when the surface form changes.

start with a simple rule:

**A implies B, and B implies C.**

now rename everything:

**Zorp implies Kelm, and Kelm implies Ruv.**

if the model learned the composition rule, the answer should transform consistently. more generally, for a structure-preserving transformation T, we would like

**M(T(x)) approximately equals T(M(x)).**

useful transformations include entity renaming, numerical substitution, translation, paraphrase, counterfactual facts, and synthetic instances with the same underlying graph.

this is stronger than asking the same question five different ways. the transformation should target a specific hypothesis. if you suspect name matching, rename entities. if you suspect answer-position memory, permute options. if you suspect a memorized arithmetic instance, change the operands while preserving difficulty.

[Yao et al.](https://aclanthology.org/2024.emnlp-main.990/) showed why this matters: benchmark contamination can cross language boundaries, evading simple text-overlap checks. their generalization-based tests altered answer choices to expose models that had absorbed translated benchmark material without learning the intended task behavior.

now play with the two axes. this is not a classifier. it is a map for keeping two kinds of evidence separate.

[[evidence-plane]]

the upper-right corner is the case people often argue about: the model probably encountered related material *and* generalizes well. there is no contradiction. it can have seen the problem and learned something reusable from it.

## what can you test with the access you actually have?

people sometimes compare a clever API prompt with a full causal training experiment as if they establish the same thing. they do not. select an access level below.

[[access-lab]]

open training runs are unusually valuable here. [Pythia](https://github.com/EleutherAI/pythia) releases data, code, and checkpoints throughout training, making before-and-after comparisons possible. if a document arrives between checkpoints, we can ask how probabilities, activations, or downstream behavior change around that event. that gets much closer to causal evidence than probing one finished opaque model.

## membership and influence are different again

even if we know D was in the corpus, we can ask which training examples actually influenced a prediction.

the ideal counterfactual is simple to state and expensive to run:

**I(z, q) = loss(M trained without z, q) - loss(M, q).**

if removing training example z changes the prediction on q, z had causal influence under that training setup. [counterfactual memorization](https://arxiv.org/abs/2112.12938) uses this leave-one-out idea to define document-specific memorization.

at modern scale, retraining once per candidate is impossible. influence and gradient methods approximate the counterfactual. [Chang et al.](https://openreview.net/forum?id=gLa96FlWwn) scaled such attribution to an 8-billion-parameter model trained on more than 160 billion tokens.

their result contains a particularly useful warning: passages that explicitly contain a fact are not always the passages with the greatest estimated causal influence on predicting it. influential examples may reinforce an entity, a relation type, or a broader prior.

so:

**textual provenance != causal influence.**

search can find where the answer was written. attribution asks what changed the model.

## and no, memorization and reasoning are not clean opposites

the usual cartoon says memorization retrieves an answer while reasoning computes one. real models appear messier.

in controlled synthetic experiments, [Reason to Rote](https://aclanthology.org/2025.emnlp-main.437/) found that models could memorize noisy labels while continuing to compute intermediate reasoning results. the memory-dependent answer was built on top of a more general computation rather than replacing it with a neat lookup table.

work on [token-level diagnosis of chain-of-thought](https://aclanthology.org/2025.emnlp-main.157/) reaches a related practical point: a reasoning trace can contain tokens associated with different memorization sources, and one bad step can propagate through the rest of the answer.

that gives us a less cinematic picture:

**input -> general computation -> memory-dependent modulation -> output**

visible reasoning is not proof that memory played no role. exact recall is not proof that no computation occurred.

## my practical protocol

if i genuinely needed to defend a claim that a model had seen some material, i would do roughly this:

- define the target first: exact document exposure, semantic exposure, benchmark contamination, memorization, or causal use
- freeze the model version, prompt, decoding settings, and query date
- build matched non-member controls from the same domain, period, length, and style
- use several signals, such as continuation, calibrated likelihood, low-probability tokens, and perturbation gaps
- aggregate over a dataset when possible and report uncertainty, thresholds, and error control
- test structure-preserving transformations separately from membership
- distinguish pretraining from fine-tuning, retrieval, tool use, and conversation context
- use weights, activations, checkpoints, or retraining only when the access level supports those claims
- phrase the conclusion at the level the evidence earns

the last line matters most. output-only behavior can support a contamination suspicion. it normally cannot reconstruct a closed model's training history with certainty.

## the short version

when a model nails a strangely familiar answer, at least four explanations remain live:

**exact recall, semantic memory, learned abstraction, active computation,**

plus mixtures of all four.

so yes, ask "have you seen that?" it is a good question.

just do not let the model's confidence, or your own pattern-matching brain, answer it alone.

## sources worth opening

- [Carlini et al., Extracting Training Data from Large Language Models](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting), USENIX Security 2021
- [Zhang et al., Counterfactual Memorization in Neural Language Models](https://arxiv.org/abs/2112.12938), NeurIPS 2023
- [Shi et al., Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789), 2023
- [Duan et al., Do Membership Inference Attacks Work on Large Language Models?](https://arxiv.org/abs/2402.07841), 2024
- [Yao et al., Data Contamination Can Cross Language Barriers](https://aclanthology.org/2024.emnlp-main.990/), EMNLP 2024
- [Chang et al., Scalable Influence and Fact Tracing for Large Language Model Pretraining](https://openreview.net/forum?id=gLa96FlWwn), ICLR 2025
- [Du et al., Reason to Rote: Rethinking Memorization in Reasoning](https://aclanthology.org/2025.emnlp-main.437/), EMNLP 2025
- [Li et al., Diagnosing Memorization in Chain-of-Thought Reasoning, One Token at a Time](https://aclanthology.org/2025.emnlp-main.157/), EMNLP 2025
- [Liu et al., PDR: A Plug-and-Play Positional Decay Framework](https://aclanthology.org/2026.acl-long.562/), ACL 2026
- [Zhang et al., Controllable Contamination Detection with Statistical Guarantees](https://aclanthology.org/2026.acl-long.1390/), ACL 2026
