# Day41: JIGSAWS — First Look (EDA)

## Objective

Day40 closed the CholecT50 series and set JIGSAWS (robotic bench-top
surgical gestures, paired with synchronized kinematics) as the next
dataset. Before any hypothesis-driven analysis, this day does plain
exploratory data analysis -- what the files actually contain, how
they're structured, what labels exist, and whether video and
kinematics agree with each other and with the dataset's own
documentation. This plays the same role Day01 played for CholecT50.

Three of JIGSAWS' four released tasks were downloaded: Suturing,
Needle Passing, and Knot Tying (the fourth release, `Experimental_setup`,
contains only cross-validation split definitions, not task data, and
is examined here for structure but not analyzed further).

## Method

[`jigsaws_eda.py`](jigsaws_eda.py) reads, per task: the `readme.txt`
(kinematic column layout and gesture vocabulary), the `meta_file_*.txt`
(skill labels), every `transcriptions/*.txt` (gesture segments), every
`kinematics/AllGestures/*.txt` (per-frame kinematic vectors), and
`ffprobe` metadata for a sample of videos, cross-checked against
kinematics frame counts for the same trials. Gesture ID numbers are
mapped to their standard names from Gao et al. (2014), "JHU-ISI Gesture
and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset
for Human Motion Modeling" (the source paper -- gesture *names* aren't
present in the data files themselves, only numeric IDs).

## Results

**Scale and subjects:**

| Task | Trials | Subjects | Gestures used | Segment length (frames) |
|---|---:|---|---:|---|
| Suturing | 39 | B-I (8) | 10 (of 15) | min 10, max 1106, mean 162 |
| Needle Passing | 28 | B-F, H-I (7, no G) | 10 (of 15) | min 17, max 2659, mean 163 |
| Knot Tying | 36 | B-I (8) | 6 (of 15) | min 18, max 883, mean 150 |

**Skill level distribution** (self-proclaimed: N=novice <10hrs,
I=intermediate 10-100hrs, E=expert >100hrs):

| Task | Novice | Intermediate | Expert | GRS range (mean) |
|---|---:|---:|---:|---|
| Suturing | 19 | 10 | 10 | 8-30 (19.1) |
| Needle Passing | 11 | 8 | 9 | 7-24 (14.3) |
| Knot Tying | 16 | 10 | 10 | 6-22 (14.4) |

**Kinematics: 76 dimensions confirmed for every trial in every task**,
matching `readme.txt` exactly: per hand (master-left, master-right,
slave-left, slave-right, 19 dims each) -- tooltip xyz (3), rotation
matrix (9), translational velocity (3), rotational velocity (3),
gripper angle (1). "Master" = surgeon's console-side manipulators,
"slave" = the robot's patient-side instruments actually moving in the
workspace.

**Video: 640x480, 30fps, mpeg4, two synchronized capture angles**
(`capture1`/`capture2` = left/right stereo endoscope views) for every
trial checked. This is a fundamentally different temporal grain than
CholecT50's 1fps sampling -- 30x denser.

**Video and kinematics frame counts don't match exactly, by a small,
variable offset.** Sampled across all three tasks: video always has
*more* frames than kinematics, by anywhere from 2 to 17 frames (e.g.
Suturing_B001: 5640 video frames vs. 5635 kinematics rows; Suturing_B002:
3462 vs. 3445). The offset isn't constant across trials, so it can't be
corrected with a single global shift -- any future work combining video
and kinematics frame-by-frame will need to handle this per-trial (most
likely by truncating to the shorter of the two, since kinematics logging
appears to start slightly after or end slightly before video capture).

**Gesture vocabulary is task-specific and highly imbalanced.** Suturing
and Needle Passing share the same 10-gesture subset (out of 15 gestures
defined across all of JIGSAWS); Knot Tying uses a disjoint 6-gesture
subset -- only G1 and G11 are shared with the other two tasks. Within
Suturing, frequency ranges from G3 "Pushing needle through tissue"
(164 occurrences) down to G10 "Loosening more suture" (4 occurrences) --
a >40x spread, sharper than CholecT50's own class imbalance in several
places (e.g. Day21's rarest instruments).

**`Experimental_setup` provides standard cross-validation splits** as
plain train/test file lists, in three flavors per task: `UserOut`
(leave-one-subject-out, the closest analogue to this project's
video-level splits throughout the CholecT50 series), `OneTrialOut`
(leave-one-trial-out), and `SuperTrialOut`. Both "Balanced" and
"unBalanced" gesture-level variants exist. These are the community's
own standard splits, used across gesture-recognition papers built on
this dataset -- worth adopting directly rather than re-deriving a new
split scheme.

## Interpretation

**JIGSAWS is a different kind of dataset from CholecT50 along three
axes that will shape everything downstream.** First, temporal density:
30fps video with per-frame kinematics is dense enough to support
genuine motion/trajectory modeling, unlike CholecT50's 1fps sampling
(which Day39 showed already helps at just a 2-second window -- JIGSAWS
offers roughly 30x that resolution for free). Second, the label
granularity is different in kind, not just in vocabulary size: JIGSAWS'
gesture segments are variable-length intervals (as short as 10 frames,
as long as 2659) marking a specific dexterous motion, closer to
temporal action segmentation than to CholecT50's per-frame multi-label
tags. Third, JIGSAWS pairs video with a second, structured modality
(76-dim kinematics) recorded from the robot itself, not derived from
the video -- an entirely different kind of signal from anything in the
CholecT50 series, which only ever had pixels and human-annotated
labels.

**The subject-level imbalance and small scale set expectations early.**
7-8 subjects per task, contributing very unevenly (self-proclaimed
skill isn't even across subjects, and gesture frequency is dominated by
a handful of common gestures) means any subject-level split (the
project's default methodology, following CholecT50's video-level split
discipline) will have very little data per fold -- a constraint to
design around from day one, not discover later the way Day01's
CholecT50 chronological-split confound was discovered after the fact.

## Reflection

Running this EDA surfaced one concrete methodological trap before any
model got built: the video/kinematics frame-count mismatch. Had a
future day assumed frame `i` of the video corresponds to row `i` of the
kinematics file without checking, any frame-level fusion of the two
modalities would have been silently misaligned by a few frames per
trial -- small, but exactly the kind of thing that degrades results in
a way that's hard to diagnose after the fact. Finding this now, in an
EDA day with no model on the line, is cheap; finding it three days into
a modeling effort would not have been.

## Direction for This Arc

Discussed and agreed with the owner before any modeling begins, so it's
recorded here rather than only in chat:

**The goal is short-horizon kinematic trajectory forecasting, not
gesture/skill classification and not pixel-level video generation.**
The owner's stated interest (surgical world models -- predicting how a
surgical scene evolves, not just labeling what's in a frame) was
checked directly against what's actually reachable on this hardware
with this dataset. Full text-conditioned video generation ("generate a
video of suturing a 3cm laceration with running suture") is not
feasible here for three independent reasons: JIGSAWS has no text
pairing at all (no wound size, technique, or scenario labels --
recordings of the same fixed bench-top drill repeated); pixel-level
video generation at usable quality requires training data and compute
several orders of magnitude beyond an 8GB RAM machine; and even with
unlimited compute, JIGSAWS' own lack of scenario variation couldn't
support conditioning on a description like that. This was ruled out
explicitly rather than quietly scaled down.

**What's actually planned**: a model that takes a short window of past
kinematics (most likely the slave-side tooltip xyz and gripper angle --
the instrument's actual position in the workspace, as opposed to the
master/console side) and predicts the next window (order of 1-2
seconds, 30-60 frames at JIGSAWS' 30Hz) as a coordinate trajectory --
not pixels, not natural language. This operates entirely in the
76-dimensional kinematic state space already present in the data,
which keeps it tractable while still being a genuine forecasting model
of surgical motion dynamics, honestly framed as a small step toward
the world-model direction rather than the thing itself. A natural
later extension: condition the prediction on the current gesture label
(e.g. G3, "pushing needle through tissue") to see whether a
gesture-aware forecaster does meaningfully better than an unconditioned
one -- but the unconditioned version comes first.

**Planned presentation**: overlaying the predicted tooltip trajectory
on the real endoscope video, next to the ground-truth continuation, so
the forecast is visually checkable rather than just a number.

*Correction, added at the start of Day42*: this specific plan turned
out not to be feasible. JIGSAWS ships no camera calibration
(intrinsic/extrinsic parameters) for either capture, and the tooltip
xyz in kinematics is in robot/world coordinates, not image coordinates
-- there is no documented way to project one onto the other correctly.
Fitting a projection by hand would risk exactly the kind of
plausible-looking-but-unjustified overlay the anti-fabrication rule
below is meant to prevent. The presentation is changed to a 3D
trajectory plot (predicted vs. ground truth, both shown together) with
no video overlay.

**An explicit anti-fabrication rule for this whole arc, raised by the
owner and binding on every future day that touches this**: a trajectory
overlay is trivial to fake -- plotting the *ground-truth* trajectory
back onto its own video will always look perfect, because it isn't a
prediction at all. Every trajectory shown as a "prediction" from this
point forward must be computed using only information available before
the prediction window (no access to the frames or kinematics being
predicted), evaluated on held-out test trials the model never trained
on, and any figure or video overlay must visually distinguish the
predicted path from the ground-truth path (e.g. different colors/line
styles, both shown together) rather than presenting one path that could
be either. This mirrors the disclosure discipline already established
for `playground/`'s toy reproductions (state plainly what was actually
run vs. simulated) and is being written down now, before the first
model exists, rather than after.

## Conclusion

JIGSAWS' three downloaded tasks (Suturing, Needle Passing, Knot Tying)
total 103 trials across 7-8 subjects each, with 30fps stereo video,
76-dimensional synchronized kinematics (with a small, variable,
per-trial frame-count offset against video that must be handled
explicitly), and gesture-segment transcriptions using a 15-gesture
vocabulary of which each task uses only a task-specific subset (10, 10,
and 6 gestures respectively). The dataset's own standard cross-validation
splits (`Experimental_setup/`, leave-one-user-out and leave-one-trial-out
variants) are available and will be adopted directly. This is a
structurally different dataset from CholecT50 -- denser in time,
paired with a genuine second modality, and labeled at the segment
rather than per-frame level. The arc's direction is now set (short-
horizon kinematic trajectory forecasting, visualized as a video overlay,
with an explicit anti-fabrication rule for every prediction shown going
forward) -- Day42 begins the first concrete model against it.
