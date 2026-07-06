Surgical Video AI Learning Journey

General surgeon exploring surgical video analysis, computer vision, and AI.

Background

* General surgeon in Japan
* Interested in surgical AI and computer vision
* Learning surgical video understanding through public datasets and research papers
* Exploring the future of AI-assisted surgery

Learning Log

Day 01 - CholecT50 Phase Timeline

* Loaded CholecT50 annotations
* Extracted phase transition points
* Calculated phase durations

Key findings:

* Surgical phases are human-defined labels imposed on a continuous process.
* Phase transitions are often difficult to determine from a single frame.
* Temporal context is essential for interpreting surgical workflow.

See [day01 details](day01_phase_timeline/README.md).

Day 02 - Triplet Exploration

* Counted instrument-verb-target (IVT) triplet frequencies in VID01.
* Most frequent triplets involve gallbladder grasping and dissection.

See [day02 details](day02_triplet_exploration/README.md).

Day 03 - Phase-Triplet Relationship

* Mapped which triplets appear within each surgical phase.
* Different phases are characterized by distinct triplet patterns.

See [day03 details](day03_phase_triplet_relation/README.md).

Day 04 - Phase Transition Exploration

* Inspected triplets around each phase transition point.
* Some transitions coincide with a new instrument appearing; others precede it, suggesting phase labels reflect workflow intent rather than a single visual event.

See [day04 details](day04_phase_transition/README.md).

Day 05 - Phase-Specific Triplets and Transition Triggers

* Looked for triplets that could serve as transition trigger candidates.
* Instrument presence alone (e.g. the irrigator) is not a reliable phase indicator.

See [day05 details](day05_phase_specific_triplets/README.md).

Day 06 - Triplet Phase Predictability

* Measured how strongly each triplet predicts a single phase (dominant phase count / total count).
* Triplets like `clipper,clip,cystic_duct` are almost perfectly phase-specific; generic ones like `grasper,grasp,gallbladder` are not.

See [day06 details](day06_triplet_phase_predictability/README.md).

Day 07 - Triplet Lifetime Analysis

* Measured how long each triplet stays active instead of just counting occurrences.
* Long-lasting triplets correspond to meaningful surgical subtasks; short-lived ones are often transient (e.g. irrigation).

See [day07 details](day07_triplet_sequence/README.md).

Day 08 - Triplet Persistence Distribution

* Computed summary statistics and a histogram of triplet lifetimes across the whole video.
* The distribution is right-skewed: median lifetime (8 frames) is far below the mean (23.5 frames).

See [day08 details](day08_triplet_persistence_distribution/README.md).

Day 09 - Triplet Recurrence Analysis

* Counted how many times each triplet reappears throughout the procedure.
* Recurrence and lifetime capture different, complementary temporal properties.

See [day09 details](day09_triplet_recurrence/README.md).

Day 10 - Frame-to-Frame Similarity Analysis

* Compared consecutive frames using Jaccard similarity of active triplet sets.
* Most transitions are stable (median similarity 1.0); low-similarity frames mark meaningful workflow changes.

See [day10 details](day10_frame_similarity/README.md).

Day 11 - Change Point Detection

* Detected frames where Jaccard similarity drops below 0.5 and exported them to CSV.
* Change points often align with instrument replacement or subtask transitions.

See [day11 details](day11_change_point_detection/README.md).

Day 12 - State Segmentation

* Compressed consecutive similar frames into "states" using a Jaccard similarity threshold.
* Bridges frame-level annotation toward representing a video as a sequence of states — a step toward Transformer-style sequence modeling.

See [day12 details](day12_state_segments/README.md).

Day 13 - State Transition Matrix

* Assigned a stable ID to each distinct state and counted transitions between consecutive states (a Markov chain).
* Generic states (idle, grasping the gallbladder) are frequent but unpredictable; task-specific states (clipping, packaging) are rare but almost always followed by the same next state.

See [day13 details](day13_state_transition/README.md).

Day 14 - Multi-Video State Vocabulary and Markov Prediction

* Scaled the state vocabulary and Markov transition model from 1 video to all 50 CholecT50 videos, with a proper train/test split (40 train / 10 test).
* The Markov model reaches 34.5% next-state accuracy on held-out videos, nearly 3x a naive baseline (12.1%) — confirming the transition patterns generalize across patients, not just one video.

See [day14 details](day14_multi_video_markov/README.md).

Day 15 - Macro vs Micro Predictability

* Compared CholecT50's own phase-level transitions (98.2% accuracy) against Day14's triplet-state transitions (34.5%), then split the triplet-state accuracy into phase-boundary-crossing vs within-phase transitions (26.3% vs 34.8%).
* The high phase-level accuracy mostly reflects that surgical phases follow a near-fixed clinical order, not a subtle model insight. Boundary-crossing transitions were the *hardest* to predict at the state level (rare, one-off events per video) — closing out the Markov-chain line of investigation: the ~35% ceiling is set by what triplet labels can express, not by memory length.

See [day15 details](day15_phase_vs_state_markov/README.md).

Next steps:

* Move from symbolic Markov chains toward representation learning: embeddings, then sequence models with more memory (RNN), then Attention
* Treat the state-sequence data as pedagogical material for learning these mechanisms, not as a benchmark to beat
* Work toward Attention and Transformer-based surgical workflow understanding
