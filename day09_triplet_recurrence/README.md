# Day09: Triplet Recurrence Analysis

## Objective

While the previous analysis focused on how long each triplet remained active (lifetime), this study investigates how many times each triplet reappears throughout an entire surgical procedure.

Recurrence represents another important temporal characteristic of surgical workflow.

---

## Method

For every frame:

- detect currently active triplets
- identify when a triplet disappears
- count each continuous appearance as one occurrence
- repeat until the end of the video

The recurrence count therefore represents the number of independent appearances of each triplet.

---

## Results

A total of **22 unique triplets** appeared during the operation, producing **109 individual action events**.

Average recurrence:

- **4.95 occurrences per triplet**

Most recurrent triplets:

| Triplet | Occurrences |
|---------|------------:|
| grasper, grasp, gallbladder | 34 |
| grasper, retract, gallbladder | 14 |
| grasper, grasp, specimen_bag | 10 |
| irrigator, aspirate, fluid | 8 |
| grasper, null_verb, null_target | 8 |

The recurrence histogram showed a highly skewed distribution.

Most triplets appeared only once or twice, whereas only a few actions were repeatedly observed throughout the operation.

---

## Observation

Triplet recurrence captures temporal behavior that is different from lifetime.

For example:

- **grasper grasp gallbladder** reappeared many times during the operation, reflecting repeated tissue manipulation.
- **clipper clip cystic duct** appeared only twice, despite being a highly phase-specific action.

From a surgical perspective, this is consistent with laparoscopic cholecystectomy.

Gallbladder grasping is repeatedly required throughout the procedure, whereas clipping of the cystic duct occurs only during a specific operative step.

Thus, recurrence reflects how frequently an action returns, rather than how long it lasts.

---

## Conclusion

Lifetime and recurrence describe different temporal properties of surgical actions.

Together, they provide complementary information for surgical workflow understanding.

These temporal characteristics may become useful features for future sequence models such as Temporal Transformers and Surgical Foundation Models.
