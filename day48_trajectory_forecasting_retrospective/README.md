# Day48: Retrospective — Closing the Trajectory-Forecasting Arc (Day42-47)

## Objective

Day41 set this JIGSAWS arc's direction: forecast a surgical
instrument's future kinematic trajectory, a small, honest step toward
the owner's stated interest in surgical world models, after ruling out
full video generation as infeasible. Six days later (Day42-47), every
angle tried -- naive baselines, a single-shot model, autoregressive
decoding, scheduled sampling, smoothness regularization, gesture
conditioning -- has produced the same underlying result: no model
achieves both real accuracy and genuine physical plausibility at once,
and the reasons why have converged rather than diverged. Following the
same discipline as Day30/35/40 in the CholecT50 series, this closes
the arc with a synthesis across all six days rather than a seventh
attempt at the same target.

## Six Days, One Task

**Day42** established naive baselines with no learning at all: last-
position-held and constant-velocity extrapolation, on Suturing's
slave-right tooltip. A clean crossover emerged -- constant-velocity
wins short-term (+0.1s: 0.46mm vs. 1.21mm) but loses badly at +1.0s
(12.57mm vs. 9.41mm) -- because real motion changes direction within a
second more often than it continues straight. This set the numeric bar
every later model had to clear.

**Day43** built the arc's first learned model: a single-shot GRU
(encode past 1s, decode the flattened future 30 frames in one linear
layer). It beat both baselines on mean displacement error (4.14mm) --
the only model in this arc to do so cleanly -- but a second metric
(frame-to-frame step size) revealed its predicted paths are over 3x
jitterier than real motion, invisible to the primary metric because
displacement error never checks the relationship between consecutive
predicted frames.

**Day44** targeted that mechanism directly: an autoregressive GRUCell
decoder, predicting one step at a time. Trained with pure teacher
forcing, evaluated free-running, it failed catastrophically (21.28mm
mean error, 51.73mm at +1.0s) -- a named risk (exposure bias) that
materialized severely, with four different held-out trials converging
to nearly the same generic curve regardless of their actual input.

**Day45** applied the standard fix: scheduled sampling, ramping
teacher-forcing probability to zero over training so train and eval
conditions match by the final epoch. This fixed the exposure bias
decisively (6.51mm mean error) and produced the arc's best smoothness
result (0.353mm step size, essentially matching ground truth's
0.388mm) -- but still didn't beat either baseline on accuracy.

**Day46** tried the other option named in Day44's Reflection: keep
Day43's accurate single-shot architecture, add a jerk penalty to the
loss instead of introducing autoregression. This achieved the arc's
best combined accuracy-smoothness trade-off (4.19mm mean error, 0.512mm
step size) with no exposure-bias risk at all -- but a third metric
(path efficiency: net displacement over total path length) showed the
predicted paths are still less than half as directionally purposeful
as real motion (0.285 vs. 0.705), prompting a fundamental question:
is unconditioned trajectory forecasting even well-posed, given the
motion is driven by a surgeon's real-time judgment not present in past
kinematics?

**Day47** tested the proposed reframing: condition on the *known*
gesture (oracle label) instead of guessing an undetermined future
choice. This made accuracy and smoothness *worse* (4.27mm, 1.013mm
step size) for a small path-efficiency gain (0.151 to 0.211, still far
from 0.691). A sharper follow-up hypothesis -- that the mechanically
constrained needle-driving motion (G3) should be far more predictable
than free-choice gestures -- also found no standout signal for G3
specifically, likely because the tracked signal is the instrument
driver's position, not the needle tip's.

**The full numeric picture, one task, six methods:**

| Method | Mean error (mm) | Step size (mm) | Path efficiency |
|---|---:|---:|---:|
| Last-position-held (Day42) | 5.11 | 0.000 | -- |
| Constant-velocity (Day42) | 5.06 | 0.395 | 1.000 (trivial) |
| Single-shot GRU (Day43) | **4.14** | 1.258 | 0.161 |
| Autoregressive, teacher-forced (Day44) | 21.28 | 1.755 | -- |
| + Scheduled sampling (Day45) | 6.51 | **0.353** | -- |
| Single-shot + jerk penalty (Day46) | 4.19 | 0.512 | 0.285 |
| + Gesture conditioning, oracle (Day47) | 4.27 | 1.013 | 0.211 |
| *Ground truth (reference)* | -- | *0.388* | *0.691* |

No row beats the baselines on accuracy AND approaches ground truth on
both smoothness and path efficiency simultaneously.

## Cross-Cutting Lessons

**1. A model can win on the metric you're checking and still be
obviously wrong, and this arc needed three independent metrics before
getting a complete picture.** Day43 needed step size to catch what
displacement error couldn't see. Day46 needed path efficiency to catch
what step size couldn't see (small, tight oscillation still isn't
purposeful motion). Each metric is a narrow, specific check; none of
them alone was sufficient evidence of a good trajectory model, and each
new metric this arc introduced existed because the previous one had
just been shown incomplete.

**2. A named risk, tested directly, can materialize exactly as badly
as predicted.** Day44's exposure-bias concern was stated explicitly
before training, and the result (4-5x worse than baselines, generic
input-independent curves) was not a subtler version of the risk -- it
was the risk, at full severity. This is the mirror image of a lesson
from the CholecT50 series (a named limitation, tested, often turns out
not to be the real one) -- here, it was.

**3. Fixing one failure mode can cost the thing that was working,
rather than adding cleanly on top of it.** Day45's scheduled sampling
fixed smoothness but sacrificed the accuracy edge Day43 had already
won; Day46's jerk penalty kept accuracy while fixing smoothness more
narrowly (step size, not path efficiency). No single change accumulated
benefits without trade-offs -- each fix targeted one specific,
previously-identified failure mode and left others (or introduced new
visibility into others) untouched.

**4. A well-motivated reframing can still fail, and that's real
information, not just a failed attempt.** Conditioning on the known
gesture was a reasonable response to Day46's concern about task
validity -- and it didn't work, not because the reasoning was flawed,
but because a 10-category gesture label is too coarse to disambiguate
motion across a 150+ frame segment. Day47's follow-up (checking
whether the mechanically-constrained sub-motion specifically is more
predictable) further narrowed this down: even the most promising
hypothesis, tested directly, didn't show the predicted signal, likely
due to a measurement gap (driver position vs. needle position) rather
than a wrong idea.

**5. Convergent evidence across independent methods is the strongest
signal this arc produced, and it points at the task, not the models.**
Six structurally different approaches -- naive extrapolation, a
single-shot network, autoregression, scheduled sampling, explicit
smoothness regularization, and intent conditioning -- all hit
recognizably the same ceiling. That consistency, across methods with
almost nothing in common architecturally, is much stronger evidence
that raw kinematic history under-determines a human surgeon's future
motion than any single day's result could have shown alone.

## What's Still Open

- **The needle tip itself, not the instrument driver, was never
  tracked or predicted.** Day47's null result for G3 may reflect this
  measurement gap rather than refute the mechanical-constraint
  intuition; JIGSAWS' kinematics don't include a direct needle-tip
  signal, so this may not be answerable with this dataset as-is.
- **Only Suturing was used across all six days.** Needle Passing and
  Knot Tying (also downloaded, Day41) were never tested -- it's
  possible one of them has a higher proportion of mechanically-
  constrained motion (e.g. Knot Tying's repetitive looping) that would
  show a cleaner predictability signal.
- **Sub-gesture temporal position (progress within a gesture segment)
  was never used as a conditioning signal**, only the coarse gesture
  category -- Day47's own diagnosis suggests this is exactly the axis
  that might matter.
- **No systematic combination of fixes was tried** (e.g. gesture
  conditioning + jerk penalty together, or scheduled sampling + jerk
  penalty) -- each day changed one thing relative to the best prior
  result, so interaction effects between fixes remain unexplored.

## Reflection

The owner named the right moment to stop: after Day47, the pattern
was "some metric moves, another doesn't, no clean win," four days
running, and that flatness was itself the signal that continuing to
tune this specific target was unlikely to produce something worth
presenting. Closing the arc here rather than trying a seventh
variation is consistent with this project's standing practice (Day30,
Day35, Day40) of treating a deliberate pause as a tool, not a failure
to push through. It's also worth naming plainly what these six days
actually produced: not a working trajectory forecaster, but a
reasonably rigorous answer to a real question (can an instrument's
future path be forecast from its past kinematics alone), obtained
by testing it from enough independent angles that the answer ("only
partially, and not more so under any of these interventions") can be
trusted rather than just asserted. That is a legitimate outcome, even
though -- as the owner put it directly -- it doesn't feel like having
built something.

## Conclusion

Six days and six structurally different approaches to forecasting a
surgical instrument's future trajectory from its past kinematic state
converge on the same result: no method tested achieves both real
accuracy (beating simple baselines) and genuine physical plausibility
(smoothness and directed, purposeful motion) at once. The single-shot
model (Day43, refined with a jerk penalty in Day46) comes closest, but
even its best form is still less than half as directionally efficient
as real motion. Conditioning on known intent (Day47) -- the most
promising reframing available -- helped only marginally and at a cost
elsewhere. This is not evidence of a solvable problem left unsolved,
but convergent evidence that forecasting a human-teleoperated
instrument's exact future path from motion history alone is close to
an information limit, not a modeling gap -- closing this arc's
unconditioned and gesture-conditioned trajectory-forecasting line. The
project moves next to a differently-shaped problem where the target
doesn't require guessing an undetermined human choice: anomaly/
deviation detection (is the current motion typical for its context) or
skill assessment (predicting GRS from interpretable motion features),
both discussed as candidates after Day46. The owner chose anomaly/
deviation detection to go first, with skill assessment held as a
later direction.
