# Day10: Frame-to-Frame Similarity Analysis

## Objective

Instead of analyzing individual triplets, investigate how the entire surgical scene changes over time.

This analysis compares consecutive frames using the Jaccard similarity of active triplet sets.

A high similarity indicates that the surgical workflow remains stable, whereas a low similarity suggests a rapid transition in surgical activities.

---

## Method

For every pair of consecutive frames:

- extract all active triplets
- ignore invalid annotations
- ignore frames containing only idle instruments (null_verb / null_target)
- compute the Jaccard similarity

\[
Similarity = \frac{|A \cap B|}{|A \cup B|}
\]

where:

- **A** = triplet set of frame *t*
- **B** = triplet set of frame *t+1*

The analysis also reports the frames with the lowest similarity and identifies which triplets disappeared and appeared.

---

## Results

### Overall Statistics

| Metric | Value |
|---------|-------:|
| Number of comparisons | 1383 |
| Mean similarity | 0.965 |
| Median similarity | 1.000 |
| Minimum similarity | 0.000 |
| Maximum similarity | 1.000 |

Most consecutive frames showed almost identical triplet sets.

---

### Lowest Similarity Examples

Examples included:

- bipolar coagulation → grasper grasp
- grasper grasp → hook coagulation
- hook coagulation → hook dissection + grasper grasp
- grasper grasp → grasper retract

These transitions correspond to abrupt changes in active surgical actions.

---

## Observation

The majority of frame-to-frame transitions were highly stable.

Only a small number of transitions produced a Jaccard similarity of zero.

Unlike the previous recurrence analysis, this approach evaluates the *entire surgical scene* rather than individual actions.

This provides a more realistic representation of workflow continuity.

From a surgical perspective, many low-similarity events correspond to a change in the primary operative task rather than a complete phase transition.

For example:

- switching from coagulation to dissection
- changing tissue manipulation from grasping to retraction
- replacing one active instrument with another

These are meaningful workflow changes while still occurring within the same surgical phase.

---

## Conclusion

Frame-to-frame similarity provides a simple quantitative measure of workflow continuity.

Rather than modeling explicit action sequences, this approach measures how much the surgical scene changes over time.

This idea forms a foundation for later studies of temporal modeling, where consecutive frames are treated as related observations instead of independent images.
