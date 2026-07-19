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

Day 16 - State Embedding from Scratch

* Replaced the Markov count table with a from-scratch numpy embedding model (embedding lookup + linear + softmax, hand-written forward/backward pass, no autograd) predicting the next triplet-state.
* Accuracy matches the Markov table almost exactly (35.2% vs 34.5%), and the learned embedding space does *not* spontaneously separate by surgical phase — because the training objective (one step ahead) never rewards phase-scale structure, only local next-state structure.

See [day16 details](day16_state_embedding/README.md).

Day 17 - State RNN from Scratch

* Replaced the one-step-back objective (Markov table / Day16 embedding) with a hand-written RNN (embedding + tanh recurrence + BPTT, no autograd) that carries hidden state across a full video.
* Accuracy clears the ~35% ceiling shared by the Markov table and Day16's embedding model, reaching 40.5% — and the RNN's hidden states visibly cluster by surgical phase (in roughly procedural order) despite phase never being a training target, confirming that the ceiling was specific to one-step-back prediction. Also documents a mode-collapse failure from too-small weight initialization, fixed with Xavier-style scaling.
* Quantified this with a linear probe: a single linear layer on the frozen hidden state recovers phase at 68.4% (vs. 29.2% baseline), confirming phase is substantially linearly encoded, not just visually suggestive in a 2D PCA plot.

See [day17 details](day17_state_rnn/README.md).

Day 18 - State Attention from Scratch

* Implemented causal self-attention from scratch (forward/backward, no autograd) as an alternative to Day17's RNN: instead of compressing history into one recurrently-updated hidden vector, every position directly attends back over all earlier states.
* A single attention layer alone (no multi-head, no feed-forward network, no stacking) underperforms the RNN — 33.1% vs. 40.5% — a negative result that motivates why full Transformer blocks need more than attention alone. A positional-encoding ablation shows much of the clean phase gradient in context vectors comes from absolute position (which correlates with phase, since surgical phases proceed in roughly fixed order), not purely from content-based attention.
* A window-size study (k-th order Markov, windowed RNN, windowed attention at k=1..10) shows the RNN's edge over the ~35% floor is not reproduced by any bounded window up to 10 states — it needs something close to the full video, matching a slow-moving "procedure progress" signal rather than a short-horizon one. Windowed attention stays flat at every k, including full history, confirming its ceiling here is architectural capacity, not context length.

See [day18 details](day18_state_attention/README.md).

Day 19 - Transformer Block from Scratch

* Implemented a full Transformer (decoder) block from scratch — multi-head attention, feed-forward network, residual connections + LayerNorm — with the backward pass verified against numerical gradients before running on real data.
* Accuracy improves over plain attention (35.3% vs. 33.1%) but still falls short of the RNN (40.5%), even though a linear probe shows the block's output encodes surgical phase just as well as the RNN's hidden state (0.685 vs. 0.684) — suggesting the remaining gap is about capturing fine-grained local dynamics, not "knowing what part of the procedure this is." Also documents an overfitting failure: more training epochs (which helped Day17's RNN) made this higher-capacity model worse on this small dataset.

See [day19 details](day19_transformer_block/README.md).

This closes out the embedding → RNN → Attention → Transformer roadmap set at Day15. Across all four mechanisms, next-state accuracy moved from 34.5% (Markov) to a ceiling that never exceeded ~40% (the RNN, still the best of the four) — more context helped a little, more architectural sophistication (attention → Transformer) recovered lost ground but never exceeded what recurrence already found. The recurring conclusion: triplet-state and phase-label representations have a ceiling no architecture change here has broken through, which is the same conclusion motivating richer, anatomy-aware representations like Murali et al.'s spatiotemporal graphs (2023, arXiv:2312.06829) — see Day18.

Day 20 - Instrument Recognition from Raw Pixels

* Started a new arc: recognizing triplets directly from raw endoscopic frames (the actual CholecT50/Rendezvous task), rather than treating triplet labels as given. First using PyTorch instead of a from-scratch implementation — CNN backprop isn't a core mechanism this project needs to internalize, and Rendezvous itself uses a pretrained CNN backbone.
* Trained a frozen ImageNet-pretrained ResNet18 + linear head (same "linear probe" pattern as Day17-19, applied to vision) on VID01's frames (the only video with local raw images) for multi-label instrument recognition, with a chronological train/test split. The model does not clearly beat a trivial train-majority baseline (macro accuracy 0.807 vs. 0.811), and three of six instrument classes have zero positive examples in the test segment — a real distribution shift from testing within one video (across time) rather than across videos (across patients), the same confound Day14 was careful to avoid for the symbolic pipeline.

See [day20 details](day20_pixel_instrument_recognition/README.md).

Day 21 - Multi-Video Instrument Recognition

* Extracted 9 more videos locally (10 total: VID01, 02, 04, 05, 06, 08, 10, 12, 13, 14, ~17,600 frames) and repeated Day20's instrument recognition with a video-level train/test split (8 train / 2 test) instead of a within-video chronological one.
* This directly fixed Day20's core problem: every instrument class now has real test examples, and macro accuracy clears the trivial baseline (0.894 vs. 0.825), with the two common instruments (grasper, hook) showing a clear, genuine win (F1 0.86, 0.68). Rare instruments (bipolar, scissors, clipper, irrigator, all under 7% prevalence) remain hard to detect (F1 0.01-0.11) — now a legible data-volume limitation rather than an artifact of a broken evaluation.

See [day21 details](day21_multi_video_instrument_recognition/README.md).

Day 22 - Verb Recognition from Raw Pixels

* Repeated Day21's exact pipeline (same 10 videos, same video-level 8/2 split, same frozen ResNet18 + linear head) for verb recognition (10 classes) instead of instrument recognition, isolating the effect of the task itself.
* Verb recognition is markedly harder (macro F1 0.192 vs. Day21's 0.302), but for two different reasons, not one: grasp/retract/null_verb looks like a genuine, likely irreducible information limit (indistinguishable in a still frame, possibly ambiguous even to human annotators), while clip/cut/aspirate/irrigate — checked directly against instrument co-occurrence, 79-95% determined by instrument identity alone — looks like an architecture gap, since today's verb classifier is fully independent of Day21's instrument signal rather than conditioned on it, much closer to how Rendezvous's actual interaction-attention modules are structured.

See [day22 details](day22_pixel_verb_recognition/README.md).

Day 23 - Instrument-Conditioned Verb Recognition

* Tested Day22's hypothesis directly: concatenated the *true* instrument multi-hot label to the frozen ResNet18 feature vector before a newly trained linear head, an oracle test of whether conditioning verb prediction on instrument identity closes the gap for tool-specific verbs.
* Macro F1 more than doubled (0.192 → 0.388), and each verb's improvement tracked its instrument→verb co-occurrence strength almost exactly (clip: 0.000 → 0.629, matching clipper→clip at 94.9%; dissect: 0.652 → 0.859, matching hook→dissect at 86.6%). Genuinely ambiguous verbs (grasp, irrigate, null_verb) barely moved or slightly worsened — confirming the earlier diagnosis that Day22's weak verb performance was partly an architecture gap, not solely a data or single-frame information limit.

See [day23 details](day23_instrument_conditioned_verb/README.md).

Day 24 - Predicted-Instrument-Conditioned Verb Recognition

* Closed Day23's oracle gap: trained a real instrument classifier, conditioned verb prediction on its *predicted* probabilities (not ground truth), and evaluated end-to-end. Cached the shared frozen ResNet18 features once instead of recomputing them for two training runs.
* Macro F1 reached 0.241 — between Day22's no-conditioning floor (0.192) and Day23's oracle ceiling (0.388), but much closer to the floor. The shortfall tracks Day21's own uneven per-instrument accuracy almost exactly: verbs tied to well-detected instruments (grasper, hook) captured a real share of the oracle gain, while verbs tied to poorly-detected ones (bipolar F1 0.106, clipper F1 0.012) captured little or none, and two verbs (grasp, coagulate) actually regressed below the no-conditioning baseline — a clean demonstration that an unreliable auxiliary signal can actively hurt, not just fail to help.

See [day24 details](day24_predicted_instrument_conditioned_verb/README.md).

Day 25 - Temporal Verb Recognition

* Tested Track 1 from Day22/23's diagnosis: an 8-frame GRU over cached ResNet18 features (Day17's from-scratch RNN mechanism, now in PyTorch, applied to real visual features instead of symbolic triplet-states) predicts verb from temporal context, isolated from instrument conditioning.
* Macro F1 improved (0.192 → 0.231), but not by fixing the intended target: grasp (the grasp-vs-retract ambiguity this was designed to resolve) stayed flat (0.434 → 0.410). Most of the gain came from an unexpected source — clip jumped from undetectable to F1 0.352 with no instrument conditioning at all, plausibly via a distinctive motion signature — while two rare verbs (coagulate, aspirate) got worse, consistent with added model capacity being a net cost with too little data.

See [day25 details](day25_temporal_verb_recognition/README.md).

Day 26 - Class-Weighted Instrument Recognition

* Started Track 2 (Day21's unsolved rare-instrument problem): kept everything identical to Day21 (same 10 videos, frozen ResNet18 + linear head, same split) except giving `BCEWithLogitsLoss` a per-instrument `pos_weight` based on training rarity.
* Macro F1 improved (0.302 → 0.378), driven mainly by clipper (F1 0.012 → 0.291, a 24x jump via much higher recall), at a real, deliberate cost to precision and overall accuracy (0.894 → 0.786, now below baseline) — a concrete instance of the precision/recall trade-off discussed after Day24. Scissors barely improved (0.054 → 0.050) despite a similar recall gain to clipper's, showing class weighting fixes a model's *willingness* to guess a rare class, not its *ability* to visually distinguish it — pointing toward backbone fine-tuning as the next lever to try.

See [day26 details](day26_class_weighted_instrument_recognition/README.md).
