# Day15: Macro vs Micro Predictability

## Objective

Day14's Markov model reached 34.5% next-state accuracy on held-out
patients — far from perfect, but far above a naive baseline (12.1%). The
open question from Day14's reflection: is that ceiling mostly about
*memory length* (only looking one step back), or mostly about
*representation* (triplet labels not capturing what actually drives a
surgeon's next move)? Today tests this directly, using the surgeon's own
hypothesis: macro-level (phase) transitions should be far easier to
predict than micro-level (triplet-state, S) transitions, because phases
follow a fixed clinical order while individual actions depend on context
a triplet label cannot see.

## Method

Two scripts, same 50 videos, same 40/10 train/test split as Day14
(`random.seed(42)`, so results are directly comparable):

- [`phase_vs_state_markov.py`](phase_vs_state_markov.py) fits and
  evaluates a 1-step Markov model on CholecT50's own **phase** labels
  instead of triplet-states.
- [`context_limits_of_markov.py`](context_limits_of_markov.py) reuses
  Day14's triplet-state Markov model, but splits its test-set accuracy
  into transitions that **cross a phase boundary** vs transitions that
  **stay within the same phase**.

## Results

**Phase-level vs triplet-state-level:**

| Level | N (test) | Accuracy | Baseline |
|---|---:|---:|---:|
| Phase (macro) | 57 | 0.982 | 0.175 |
| Triplet-state S (micro, Day14) | 1423 | 0.345 | 0.121 |

**Triplet-state (S) accuracy, split by phase-boundary status:**

| Group | N (test) | Accuracy |
|---|---:|---:|
| Overall (= Day14) | 1423 | 0.345 |
| Crosses a phase boundary | 57 | 0.263 |
| Stays within the same phase | 1366 | 0.348 |

(57 boundary-crossing S-transitions exactly matches the 57 phase
transitions above — every phase change corresponds to exactly one
boundary-crossing S-transition, a useful internal consistency check.)

## Interpretation

The phase-level result (98.2%) is not a sign that the model learned
something subtle. Laparoscopic cholecystectomy phases proceed in an
almost fixed clinical order, so "predict the next phase" is close to a
solved, low-entropy problem by construction — this matches what was
already observed back in Day04/05 (specific instruments like the clipper
reliably announce a phase change).

More informative is that **boundary-crossing S-transitions are the
*least* predictable of all** (26.3%, below even the 34.5% overall
average). Three factors compound here, not one:

1. **Sample size.** Each video crosses a given phase boundary once, so
   there are far fewer training examples of any specific boundary
   transition than of routine, repeated within-phase transitions.
2. **The task itself changes shape.** Predicting "a clipper will appear
   soon" (a single attribute) is a much easier problem than predicting
   the exact next state S — a full triplet-set that must match exactly,
   including every other instrument active at the same moment. Moving
   from phase-level to S-level isn't just "more memory would help", it's
   evaluating a strictly harder, compound target.
3. **Within-phase accuracy is partly a frequency effect.** States that
   recur often within a phase (the same routine grasp, the same ongoing
   retraction) give the model many training examples of "what usually
   comes next" — so the Markov model's ~35% is close to what you'd get
   by just answering "whatever S is most common right now", not evidence
   that the model is deducing something from context.

## Reflection

The original hypothesis (macro is easy, micro is hard) was confirmed,
but in an unsurprising way for the macro side — phase order is close to
deterministic by clinical design, so 98.2% accuracy is not a discovery in
itself. What is more informative is that the boundary-crossing
transitions, despite being the moments most tied to a fixed procedural
order, were the *hardest* to predict at the S level: they are rare,
one-off events per video, so there is little repeated pattern for a
same-order model to learn from a set of other patients. Meanwhile the
"easier" within-phase transitions score better mostly because common
states are, definitionally, common — not because the model understood
why one routine action follows another. Taken together, this points to
the same conclusion reached before: predicting the next S from the
current S alone is fundamentally limited, whether or not a phase boundary
is involved, because what actually determines a surgeon's next move is
context (bleeding, tension, anatomy) that neither the phase label nor the
current triplet-state can see.

## Conclusion

Continuing to refine a Markov model over symbolic triplet-states — more
history, smarter smoothing — has limited room left to improve, because
the ceiling here is set by what the representation can express, not by
how much of the past the model is allowed to look at. This closes out the
Markov-chain line of investigation started in Day13.
