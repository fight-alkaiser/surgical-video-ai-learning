Day08: Triplet Persistence Distribution

Objective

Analyze the distribution of triplet lifetimes during laparoscopic cholecystectomy.

Rather than focusing only on representative long-lasting events, this analysis investigates how long all triplet events persist and whether most surgical actions are transient or sustained.

Understanding the temporal characteristics of triplets is an important step toward surgical workflow understanding, since temporal persistence may provide stronger cues than single-frame observations.

⸻

Method

For each triplet event:

* detect when the triplet first appears
* measure how many consecutive frames it remains active
* record its lifetime
* calculate summary statistics
* visualize the lifetime distribution using a histogram

The following statistics were computed:

* Mean
* Median
* Minimum
* Maximum
* Standard deviation

In addition, the percentage of triplets lasting no longer than 5, 10, 20, 30, and 60 frames was calculated.

⸻

Results

Lifetime Statistics

Metric	Value
Number of triplet events	109
Mean lifetime	23.49 frames
Median lifetime	8.00 frames
Minimum	1 frame
Maximum	238 frames
Standard deviation	41.36 frames

Longest Triplet Events

Triplet	Lifetime
grasper, retract, liver	238
hook, null_verb, null_target	231
hook, dissect, gallbladder	196
hook, dissect, gallbladder	142
grasper, grasp, gallbladder	98
hook, dissect, gallbladder	97
grasper, grasp, gallbladder	95
grasper, retract, liver	74
grasper, grasp, gallbladder	71
bipolar, coagulate, blood_vessel	66

Lifetime Summary

Lifetime	Percentage
≤5 frames	36.7%
≤10 frames	54.1%
≤20 frames	71.6%
≤30 frames	80.7%
≤60 frames	90.8%

⸻

Observation

The lifetime distribution is strongly right-skewed.

Although the mean lifetime was 23.49 frames, the median was only 8 frames, indicating that a small number of long-lasting events substantially increased the average.

Approximately 72% of all triplet events lasted no longer than 20 frames, suggesting that most surgical actions are relatively short-lived.

In contrast, several task-specific triplets persisted for long periods, including:

* grasper retracting the liver
* hook dissecting the gallbladder
* bipolar coagulating blood vessels

These sustained actions correspond to stable surgical subtasks rather than transient events.

⸻

Conclusion

Most triplet events are brief, while only a limited number persist over long durations.

This finding suggests that temporal persistence is an informative feature for surgical workflow understanding.

Rather than treating all detected triplets equally, future workflow recognition models may benefit from emphasizing long-lasting actions that better represent stable surgical states.
