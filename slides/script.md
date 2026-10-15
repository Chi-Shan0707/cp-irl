# Talk script: *Calibrating What Behavior Can Identify*

NeurIPS 2026 Workshop PUDM · video talk, about 4 minutes (limit 5) · Yuhan Chi

This script stands on its own: you can read it aloud without the paper or the
slides. The deck is [`slides/index.html`](index.html), also online at
<https://chi-shan0707.github.io/cp-irl/slides/>.

- **[click]** = press → or Space once. Each click reveals the next build on the
  current slide. A click after the last build moves to the next slide.
- Spoken text is in the quote blocks: about 575 words, roughly 3:50 at a calm
  150 words per minute, inside the 4-minute target and the 5-minute limit.
- Formulas are written in the text as they should be **said**. The original
  notation follows in ⟨angle brackets⟩ so you can match it to the slide.

---

## Pronunciation key

Read symbols as words, and use the same reading every time.

| On the slide | Say | Meaning |
|---|---|---|
| IRL | "eye-ar-ell" (first time: "inverse reinforcement learning, or IRL") | learning a reward from behavior |
| CIO | "see-eye-oh" (first time: "conformal inverse optimization, or CIO") | the method we build on (Lin, Delage & Chan, NeurIPS 2024) |
| i.i.d. | "eye-eye-dee" | independent and identically distributed |
| π | "pie" | a policy, i.e. a rule for choosing actions |
| *v* | "vee" | a reward difference |
| *V<sub>v</sub>*(π) | "V-sub-vee of pie" | the value of policy π under reward *v* |
| ‖*v*‖<sub>D</sub> | "the D-norm of vee" | the spread of policy values that *v* creates |
| max<sub>π</sub> … − min<sub>π</sub> … | "the best … over all policies, minus the worst" | |
| γ | "GAM-uh" | target coverage level, here 0.8, i.e. 80% |
| *q* | "cue" | calibrated radius |
| η | "AY-tuh" (US "EE-tuh" is also fine; just be consistent) | certified width of the explaining rewards |
| *R* | "are" | radius of the enlarged set |
| *R* = 2*q* + η | "R equals two-cue plus eta" | |
| ≤ *q*, ≤ η | "at most cue", "at most eta" | |
| Pr[ … ] ≥ γ | "the probability that … is at least gamma" | |
| *R* ≥ 1 | "R at least one" | |
| 24.3° | "twenty-four point three degrees" | |
| ≈ 0, 2×10⁻⁸ | "essentially zero", "two times ten to the minus eight" | |
| 0.82, 0.44 | "eighty-two percent", "forty-four percent" | |
| 0.78, 0.11 | "zero point seven eight", "zero point one one" | |
| ≠ | "is not" | |

---

## 1 · Title

> Hi, I'm Yuhan Chi. This work asks what a conformal guarantee can actually
> tell us when we learn rewards from behavior.

## 2 · Many rewards explain the same behavior

> Inverse reinforcement learning, or IRL, infers a reward from demonstrations.
> But this route is explained **[click]** by caring about speed, **[click]** by
> avoiding hazards, **[click]** or by any mix of the two. **[click]** Standard
> IRL picks one and optimizes it as if it were true.

## 3 · Background: keep a set of plausible objectives

> We start from conformal inverse optimization, or CIO. Inverse optimization is
> IRL's one-step cousin. **[click]** A single estimated objective can make a
> whole edge look optimal, so it may recommend the far corner. **[click]** CIO
> keeps a set of plausible objectives, the green wedge, and decides for the
> worst case over it. That brings the decision back to the true optimum.

## 4 · Background: how big a set?

> How big should the set be? CIO lets held-out data decide. A set *explains* a
> decision if some objective in it makes that decision optimal. Grow the set,
> with radius cue ⟨*q*⟩, until it explains at least gamma ⟨γ⟩, say eighty
> percent, of the held-out decision makers. Conformal prediction then
> guarantees it explains the next one with probability at least gamma
> ⟨≥ γ⟩. **[click]** In IRL, each decision maker is one demonstrator. Most of
> the pipeline carries over. The distance score and the robust decision must be
> rebuilt.

## 5 · Rebuilt 1: the distance score

> First, the distance score. CIO measures an angle between reward vectors. But
> stretching the features changes no reward value and no policy, yet the
> calibrated angular radius moves by twenty-four point three degrees
> ⟨24.3°⟩ on average. **[click]** Ours stays put: essentially zero.
> **[click]** We measure a reward difference vee ⟨*v*⟩ by the spread of policy
> values it creates. The D-norm of vee is the best policy value under vee,
> minus the worst ⟨‖*v*‖<sub>D</sub> = max<sub>π</sub> *V<sub>v</sub>*(π) −
> min<sub>π</sub> *V<sub>v</sub>*(π)⟩. It is the smallest distance that
> bounds regret, and the score is still one linear program.

## 6 · Touching is not containing

> Second, what the guarantee actually says. The gray wedge holds every reward
> that explains a new demonstrator. The star is the reward they actually use,
> their *true reward*. **[click]** With probability at least eighty percent,
> the calibrated set touches the wedge, **[click]** but here it misses the star.
> **[click]** At an eighty-percent target we measure eighty-two percent for
> touching and only forty-four for containing ⟨0.82 vs 0.44⟩. **[click]** The
> gap comes from which distance is scored, so more data won't close it. CIO
> already states this touching guarantee. The question is what it buys.

## 7 · Pay for the gap: R = 2q + η

> To contain the true reward, we pay for the gap explicitly. **[click]** Some
> explaining reward is within cue of the estimate ⟨≤ *q*⟩. **[click]** Rescaling
> it to unit scale costs at most cue more ⟨≤ *q*⟩. **[click]** Moving across
> the explaining rewards costs at most eta ⟨≤ η⟩, their certified width.
> **[click]** So, with R equal to two-cue plus eta ⟨*R* = 2*q* + η⟩, the
> probability that the true reward lies within R is at least gamma
> ⟨Pr[ … ] ≥ γ⟩.

*(If asked: R is capped at two, the largest possible distance; eta adapts
CIO's diameter assumption; the probability is marginal, averaged over
demonstrators.)*

## 8 · Rebuilt 2: decide against a fixed baseline

> Finally, we choose the policy with the best worst-case gain over a baseline,
> a reference policy fixed in advance. If the set contains the true reward, we
> never do worse than that baseline. **[click]** But bigger sets mean smaller
> gains. Once R is at least one ⟨*R* ≥ 1⟩, the set contains the zero reward,
> every policy ties, and the gain is zero. **[click]** Without a width
> certificate, eta is two ⟨η = 2⟩, and the method safely returns the baseline.
> **[click]** In a small toy where eta is computed exactly, R is zero point
> seven eight and the certified gain is zero point one one.

## 9 · Three separate answers

> So any calibrate-then-decide pipeline, whenever behavior only partially
> identifies what we want, has to answer three separate questions: which
> geometry, which quantile, and which event. Thanks for listening.

---

### Delivery notes

- **Credit CIO.** Say "CIO already states this touching guarantee" on slide 6
  without hedging. It is the honest framing of what is inherited.
- **Stress the scope of slide 8.** The toy result (R = 0.78, gain 0.11) is one
  exactly solved example added after acceptance, not a benchmark result.
- **Pace.** Pause briefly after each click on slide 7, where the derivation of
  R = 2q + η is built up step by step.
