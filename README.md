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

Day 27 - Backbone Fine-Tuning

* Tested Day21's other named fix: unfroze ResNet18's last residual block (layer4) and fine-tuned it, keeping Day26's class-weighted loss constant, isolating whether better features (not just loss re-weighting) fix rare-instrument detection. 8GB RAM ruled out Day24-26's feature-caching shortcut, so training went back to a live loop like Day21's, at a smaller batch size.
* Macro F1 rose from 0.378 (Day26) to 0.512, and macro accuracy from 0.786 to 0.932 — precision and recall improved *together* for nearly every instrument (clipper: precision 0.190→0.481, recall 0.629→0.503), unlike Day26's pure re-weighting which traded one for the other. Scissors remained the weakest (F1 0.101), improving only modestly despite the same intervention that transformed bipolar (0.182→0.431) and clipper (0.291→0.492) — consistent with its scarcity (~500 total instances) being severe enough that better features alone can't fully compensate.

See [day27 details](day27_backbone_finetuning/README.md).

Day 28 - Fine-Tuned Instrument-Conditioned Verb Recognition

* Combined Day24 (verb conditioned on predicted instrument probabilities) with Day27 (fine-tuned instrument classifier) to test whether a better upstream instrument signal recovers more of Day23's oracle ceiling (F1 0.388) than Day24's frozen-feature version (F1 0.241) did.
* Macro F1 reached 0.299 — recovering ~55% of the gap to the oracle ceiling, more than double Day24's ~25%. Gains concentrated exactly where Day27's fine-tuning most improved instrument detection (coagulate: 0.023→0.369; clip: 0.119→0.377), while verbs tied to instruments fine-tuning couldn't fully fix (cut, needs scissors) or with an inherently split verb profile (aspirate, needs irrigator) stayed weak or regressed — closing the loop that most of the tool-specific verb problem traces back to instrument recognition quality.

See [day28 details](day28_finetuned_instrument_conditioned_verb/README.md).

Day 29 - Target Recognition

* A deliberately lighter single day (not a new multi-day arc): applied the best known recipe directly (Day27's fine-tuned backbone + Day26's class-weighted loss) to target recognition (15 anatomical-structure classes), the third and last part of CholecT50's triplet.
* Macro F1 reached only 0.207 — below instrument's 0.512 and verb's best combined result of 0.299 (Day28) — with 6 of 15 classes completely undetected. This is a scale lesson, not a new technique-level one: target has more than double instrument's class count and several classes rarer than any instrument/verb seen so far, showing the same fixes (more data, fine-tuning, class weighting) have real but bounded power that depends on how much data exists per class. Closes the instrument-verb-target arc (Day20-29).

See [day29 details](day29_target_recognition/README.md).

Day 30 - Retrospective

* A deliberate pause between the two closed arcs (Day01-19 symbolic sequence modeling, Day20-29 pixel-based recognition) and Day31's move to self-supervised learning: not a re-summary of each day, but the methodological lessons that reappeared across both arcs in different disguises.
* Nine cross-cutting lessons identified, spanning both arcs: accuracy is close to meaningless under class imbalance; evaluation protocol can matter as much as architecture; a falling loss curve isn't proof of useful learning; representational capability must be checked (e.g. via linear probes), not assumed; "willingness to guess" and "ability to distinguish" need different fixes; errors propagate through pipelines; fixes don't always land where intended; trade-offs should be chosen deliberately, not absorbed silently; and honesty about what an experiment can't show is itself a finding.

See [day30 details](day30_retrospective/README.md).

Day 31 - Contrastive Self-Supervised Pretraining

* Opened the self-supervised learning arc: a small, resource-constrained reproduction of SimCLR-style contrastive learning (batch size 32-64, vs. hundreds-to-thousands in published results) adapting only ResNet18's layer4 on the 8 training videos' frames — no instrument/verb/target/phase label used anywhere during pretraining. Evaluated the same way as every prior instrument day: freeze the backbone, cache features, class-weighted linear probe.
* Macro F1 reached 0.407 — exactly 50% of the way from Day21's frozen-ImageNet baseline (0.302) to Day27's supervised fine-tuning result (0.512), recovered with zero labels. The recovery wasn't uniform: concentrated in instruments Day26/27 diagnosed as feature-quality-limited (bipolar, clipper: ~60% of their gap closed) and nearly absent where frozen features were already strong (grasper: 4%) — a pattern that argues the contrastive objective learned something real about this domain, not a spurious uniform boost.

See [day31 details](day31_contrastive_pretraining/README.md).

Day 32 - What's Inside the Contrastively-Learned Representation?

* Reused Day16-19's linear-probe + PCA methodology to check what Day31's label-free, temporally-blind contrastive backbone actually organizes itself around: does phase structure (never a training target, and the pretext task had no temporal signal at all) show up, and more than in plain frozen ImageNet features?
* Phase-probe accuracy improved modestly (0.511 → 0.532, baseline 0.382) — a much smaller relative gain than instrument recognition's (Day31: 0.302 → 0.407), since generic features already captured much of phase's visual correlates. The PCA plot showed this gain wasn't diffuse: a sharply-separated cluster emerged for clipping-and-cutting, the phase most tied to the clipper instrument — exactly the instrument contrastive adaptation improved most in Day31 (F1 0.012 → 0.298), connecting two independent probes to the same underlying mechanism.

See [day32 details](day32_representation_analysis/README.md).

Day 33 - Does a Bigger Batch Size Actually Help?

* Tested Day31's own named limitation directly: doubled the contrastive pretraining batch size (32→64 images, 64→128 augmented views), with learning rate scaled 2x to match (linear scaling rule), otherwise identical setup.
* Macro F1 was unchanged (0.407 → 0.406) — a clean negative result. Batch size, the most obvious hardware-constrained caveat relative to published SimCLR results, is not the binding constraint here, at least within the doubling achievable on 8GB RAM. Per-instrument changes were small and bidirectional (no consistent improvement pattern), unlike Day31's clear, uniform-looking gains over frozen ImageNet — the signature of noise rather than a real effect. Redirects the next investigation (if this arc continues) toward training duration, augmentation choices, or a data-diversity ceiling instead.

See [day33 details](day33_batch_size_ablation/README.md).

Day 34 - Temporal-Order Self-Supervised Pretraining

* Tried a genuinely different self-supervised signal that uses temporal structure (unlike Day31-33's appearance-only contrastive learning): predict which of two same-video frames comes later, via a single ranking-loss-trained "progress head," still using zero labels. Hypothesis: this should capture phase structure better than contrastive learning did.
* Split verdict. Instrument recognition improved to macro F1 0.432 — the best SSL result so far, beating both contrastive variants. But phase-probe accuracy *fell* to 0.460, worse than even plain frozen ImageNet (0.511), directly contradicting the hypothesis. A diagnostic (PCA colored by video vs. phase) didn't cleanly confirm the leading explanation (within-video overfitting) and surfaced a broader methodological point instead: a 2D PCA plot's apparent structure and a full-dimensional linear probe's actual accuracy can diverge — only the probe answers what a linear classifier can actually use.

See [day34 details](day34_temporal_order_pretraining/README.md).

Day 35 - Retrospective: the Self-Supervised Learning Arc (Day31-34)

* A deliberate pause (same discipline as Day30) to synthesize what generalized across four days and three pretext tasks (appearance contrastive, contrastive at 2x batch size, temporal order) against two downstream probes (instrument, phase), rather than re-summarizing each day.
* Four cross-cutting lessons: no pretext task is free of an inductive bias that automatically aligns with every downstream task (temporal-order won on instrument, lost on phase); a named limitation (batch size) tested directly turned out not to be the real one, again; independent probes agreeing (Day31 instrument F1 + Day32 phase PCA, both pointing at clipper) is stronger evidence than either alone, and disagreeing (Day34's instrument-vs-phase split) is just as informative; a 2D PCA plot's apparent structure and a linear probe's actual accuracy can diverge. Reframes the arc's question from "does SSL help" to "which inductive bias, for which downstream question."

See [day35 details](day35_ssl_retrospective/README.md).

Day 36 - Extending SSL Evaluation to Verb and Target

* Reused the exact backbones already saved from Day31 (contrastive) and Day34 (temporal-order), no retraining, and evaluated them on verb (10 classes) and target (15 classes) with the same class-weighted linear probe recipe — extending the instrument/phase picture to four tasks total.
* Neither SSL method helped verb at all (0.304–0.309 across all three backbones, indistinguishable from noise); target showed a small, one-sided effect (temporal-order 0.220 vs. frozen 0.209, contrastive flat). A side comparison showed class-weighting alone (no SSL) already lifts verb from 0.192 to 0.309, generalizing Day26's instrument finding. Reading all four tasks together: SSL adaptation helps in proportion to how much of a task's difficulty is genuinely a feature-separability problem (instrument) versus a single-frame information limit, architectural gap, or data-scarcity problem (verb, target) — diagnoses Day22/23/29 had already made independently, before any SSL evaluation existed.

See [day36 details](day36_verb_target_ssl_evaluation/README.md).

Day 37 - Testing the Verb Recognition Architecture Gap

* Day22 diagnosed verb's low ceiling as split between a single-frame information limit and an architecture gap (the model never sees instrument identity, even though verb meaning is instrument-dependent). Day31-36 ruled out feature quality as the fix; today tests the architecture-gap half directly, on frozen ImageNet features, by concatenating instrument information onto the verb probe's input: ground-truth instrument one-hot (oracle) vs. predicted instrument probabilities from a separately trained probe (realistic).
* Oracle conditioning lifted verb macro F1 from 0.309 to 0.484, concentrated on instrument-specific verbs (clip 0.351→0.701, cut 0.061→0.444, coagulate 0.287→0.616, dissect 0.701→0.893) — confirming the architecture-gap hypothesis. But realistic conditioning recovered none of it (0.305, indistinguishable from baseline), because the instrument probe supplying the signal is itself only macro F1 0.399 — its errors are correlated with the same visual ambiguity that makes verb hard. The two problems are coupled: fixing verb via instrument conditioning is gated on first fixing rare-instrument recognition, which SSL adaptation has already been shown not to solve.

See [day37 details](day37_verb_instrument_conditioning/README.md).

Day 38 - Does a More Accurate Instrument Predictor Close More of the Gap?

* Day37 explained its oracle-realistic gap (ground-truth instrument conditioning: 0.309→0.484; predicted-instrument conditioning: 0.305, no gain) as the predictor being too noisy (instrument macro F1 only 0.399). Today re-runs Day27's fine-tuning recipe (saving the checkpoint this time, which Day27 didn't) to get a substantially more accurate instrument predictor (macro F1 0.512) and swaps it into the same realistic-conditioning setup.
* No improvement: verb macro F1 stayed at 0.305, identical to Day37's weaker predictor, despite large real gains on bipolar and clipper (the instruments coagulate and clip most depend on). This rules out predictor accuracy as the limiting factor and points to a structural, frame-level explanation instead — verb difficulty and instrument-prediction difficulty likely share a cause (plausibly instrument-tip occlusion), so a predictor that's more accurate on average still fails on exactly the frames where its help is needed most.

See [day38 details](day38_finetuned_instrument_verb_conditioning/README.md).

Day 39 - Does Temporal Context Help Instrument or Verb More?

* Day38 pointed to a structural, frame-level explanation for verb's oracle-realistic gap; the owner proposed instrument-tip occlusion as the shared cause, predicting temporal context should help instrument (trackable across frames) more than verb (which needs the occluded instant itself). Today tests this with a 3-frame window (t-1, t, t+1, ~2 seconds at CholecT50's 1-second sampling) on frozen ImageNet features, comparing macro F1 against single-frame on the identical frame subset, for both tasks.
* The hypothesis was not supported: verb improved 2.5x more than instrument (+0.040 vs. +0.015), concentrated in rare, brief verbs (coagulate, clip, pack). Precision/recall analysis shows the mechanism is the opposite of "more willingness to guess" — precision rises sharply while recall falls, meaning temporal context lets the model correctly reject single-frame false positives on transient actions, closer to Day27's "genuinely better features" signature than Day26's "willingness" signature. A notable closing number: this untrained 3-frame concatenation (verb macro F1 0.332) beats every SSL-pretrained backbone from Day36 (0.304-0.309) on the same task.

See [day39 details](day39_temporal_context_instrument_verb/README.md).

Day 40 - Retrospective: Closing the CholecT50 Series (Day01-39)

* The project-level retrospective closing all four arcs (symbolic sequence modeling, supervised pixel recognition, self-supervised learning, and the Day37-39 verb-architecture investigation) before moving to a new dataset. Its sharpest finding only became visible by comparing across arcs: Day28's fully-conditioned, fine-tuned-instrument verb model (F1 0.299, from the Day20-29 arc) scores *below* Day36's plain class-weighted baseline with no conditioning at all (F1 0.309) — a loss-function fix and an architecture fix were targeting the same underlying failure mode and don't stack, while Day39's temporal context (a different failure mode) added real value on top.
* Also includes a Clinical Implications section: the class-uniform macro-F1 metric this whole series optimized doesn't reflect what would actually matter in practice — a `grasp`/`retract` mix-up is closer to an interpretation difference than an error, while a `clip`/`cut` mistake at a safety-critical moment is a fundamentally different kind of failure the metric doesn't distinguish. Real intraoperative usefulness would look like hazard/next-step awareness (a driver-assistance model), not finer-grained instrument/verb/target labeling — a different task this series never built or evaluated toward. Closes the CholecT50 series; the project moves next to JIGSAWS (robotic bench-top gestures + synchronized kinematics).

See [day40 details](day40_cholect50_series_retrospective/README.md).

JIGSAWS (robotic bench-top surgical gestures + synchronized kinematics)

Day 41 - JIGSAWS: First Look (EDA)

* Plain exploratory data analysis on the three downloaded JIGSAWS tasks (Suturing, Needle Passing, Knot Tying) before any modeling: 103 trials across 7-8 subjects, 30fps stereo video, 76-dim synchronized kinematics (master/slave tooltip xyz, rotation, velocity, gripper angle), and gesture-segment transcriptions from a 15-gesture vocabulary (each task uses only a task-specific subset). Found a small, variable per-trial frame-count offset between video and kinematics that must be handled explicitly rather than assumed away, and confirmed the dataset's own standard cross-validation splits (`Experimental_setup/`, leave-one-user-out and leave-one-trial-out) are usable directly.
* Sets the arc's direction after discussion: short-horizon kinematic trajectory forecasting (predict ~1-2s of future tooltip position from a past window, in the 76-dim kinematic state space, not pixels), presented as a video overlay of predicted-vs-actual tooltip path — a small, honest step toward the owner's stated interest in surgical world models, after explicitly ruling out full text-conditioned video generation as infeasible on this hardware and with this dataset. Includes a binding anti-fabrication rule for every future day in this arc: any trajectory shown as a "prediction" must use only past information, be evaluated on held-out trials, and be visually distinguished from ground truth.

See [day41 details](day41_jigsaws_eda/README.md).

Day 42 - Trajectory Forecasting: Naive Baselines

* Before any learned model, established what zero-training baselines achieve on the arc's core task: forecasting a surgical instrument's future position from its past kinematic state (Suturing, slave-right tooltip, 1s past predicting 1s ahead, 4,319 windows across 39 trials). Two closed-form baselines compared: last-position-held (predict no movement) and constant-velocity (extrapolate using the robot's own recorded velocity), both using only information from before the prediction start per Day41's anti-fabrication rule.
* Found a clean crossover: constant-velocity wins at short horizons (0.46mm at +0.1s vs. 1.21mm) but loses badly at longer ones (12.57mm at +1.0s vs. 9.46mm), because real surgical motion changes direction within a second far more often than it continues straight. This gives any future learned model two concrete numbers to beat at two different horizons, and previews the design question for Day43: can a model combine short-term momentum with longer-term awareness of upcoming direction change?

See [day42 details](day42_trajectory_forecasting_baseline/README.md).

Day 43 - A GRU Trajectory Model: Wins on the Metric, Fails on Physical Plausibility

* Built the arc's first learned model: a single GRU encoding the past 1s of slave-right tooltip kinematics, decoded in one shot to the future 30-frame position sequence, trained/evaluated with leave-one-subject-out (reproducing JIGSAWS' official UserOut split) and Day42's baselines recomputed on the same held-out subject for a fair comparison.
* The GRU beats both baselines on mean displacement error (4.14mm vs. 5.11mm/5.06mm) and at every horizon checkpoint from +0.3s onward — a real win on the target metric. But an independent smoothness check (frame-to-frame step size, not compared to ground truth at all) shows the model's predicted path moves over 3x farther between consecutive frames than real tooltip motion ever does — a physically implausible, jittery trajectory the displacement metric can't detect, because a single-shot linear decoder has no constraint linking one predicted frame to the next. Sets up Day44 to fix this specific failure mode (autoregressive decoding or a smoothness-penalized loss) rather than treating the metric win as the finish line.

See [day43 details](day43_gru_trajectory_model/README.md).

Day 44 - Autoregressive Decoding: The Named Risk Materializes, Badly

* Tested the structural fix aimed directly at Day43's jaggedness problem: a GRUCell decoder predicting one step at a time, chained through the recurrence, trained with teacher forcing and evaluated with free-running rollout (the model sees only its own predictions, matching real use). The exposure-bias risk was named explicitly before running anything.
* The risk dominated everything: the autoregressive model is 4-5x worse than Day42's baselines (mean error 21.28mm vs. 5.06-5.11mm; 51.73mm vs. 9.41-11.12mm at +1.0s), despite near-zero teacher-forced training loss. The example plots show something more specific than plain drift — four different held-out trials converge to nearly the same generic curve regardless of their actual input, suggesting the free-running decoder's dynamics are dominated by the GRUCell's own attractor behavior rather than the encoder's context. Day43's flawed single-shot model remains this arc's only result beating both baselines; Day45 tries scheduled sampling to fix exposure bias without abandoning autoregression.

See [day44 details](day44_autoregressive_trajectory_model/README.md).

Day 45 - Scheduled Sampling: Smoothness Fixed, Accuracy Still Not There

* Applied the standard fix for Day44's exposure bias: scheduled sampling, ramping the probability of feeding the decoder its own prediction (instead of the true delta) linearly from 0 to 1.0 over training, so training conditions match free-running evaluation by the final epoch. Same architecture, held-out subject, and baselines as Day44.
* Mean error dropped 3.3x (21.28mm to 6.51mm) and predicted-path smoothness now essentially matches real tooltip motion (0.353mm step size vs. ground truth's 0.388mm) — the best smoothness result in the arc. But the model still doesn't beat either of Day42's baselines on displacement error, and remains behind Day43's jagged-but-accurate single-shot model. Three learned models in, no single one has been both accurate and physically plausible at once. Also records a caveat raised by the owner: every method tried shows the same ~10x error growth from +0.1s to +1.0s, suggesting a 1-second horizon is close to trajectory data's actual predictability ceiling (driven by the surgeon's judgment, not encoded in past kinematics) rather than a gap a smarter model will simply close.

See [day45 details](day45_scheduled_sampling/README.md).
