# Day07: Triplet Lifetime Analysis

## Objective

Instead of counting how often triplets appear, investigate how long each surgical action remains active.

Persistent actions may represent stable workflow states, whereas short-lived actions may correspond to transient events or annotation noise.

---

## Method

For every frame:

- detect newly appearing triplets
- record the first frame
- detect when the triplet disappears
- calculate its lifetime

Only triplets lasting at least **20 frames** were reported.

---

## Representative Results

| Triplet | Duration (frames) |
|---------|------------------:|
| hook, null_verb, null_target | 231 |
| hook, dissect, gallbladder | 196 |
| hook, dissect, gallbladder | 142 |
| grasper, retract, liver | 238 |
| clipper, clip, cystic_duct | 49 |
| bipolar, coagulate, blood_vessel | 66 |
| grasper, grasp, specimen_bag | 40 |

---

## Observation

Several task-specific triplets persisted for relatively long periods.

Examples include:

- hook dissecting the gallbladder
- bipolar coagulation
- clipper clipping the cystic duct
- grasper retracting the liver

These sustained actions correspond well to meaningful surgical subtasks.

In contrast, many irrigation-related triplets appeared for only one or two frames before disappearing.

This suggests that not every detected action represents a stable workflow state.

From a surgical perspective, this is expected.

Irrigation is frequently used throughout laparoscopic cholecystectomy for:

- smoke removal
- visualization
- aspiration of small amounts of blood
- cleaning the operative field

Therefore, irrigation events are intermittent by nature and should not be interpreted as phase boundaries.

---

## Conclusion

Triplet duration provides additional temporal information beyond simple occurrence counts.

Long-lasting task-specific actions may provide stronger cues for surgical workflow understanding than brief instrument appearances.
