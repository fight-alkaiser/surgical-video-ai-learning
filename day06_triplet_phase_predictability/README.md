Day06: Triplet Phase Predictability

Objective

Investigate how strongly individual triplets predict surgical phases in CholecT50 VID01.

Method

For each triplet, count its occurrences across all phases.

Calculate:

dominant phase count / total occurrence count

A value close to 1.0 indicates that the triplet is highly phase-specific.

Results

Most Phase-Specific Triplets

Ratio	Count	Dominant Phase	Triplet
1.00	100	carlot-triangle-dissection	bipolar,coagulate,blood_vessel
1.00	78	clipping-and-cutting	clipper,clip,cystic_duct
1.00	27	gallbladder-packaging	grasper,pack,gallbladder
1.00	24	clipping-and-cutting	clipper,null_verb,null_target
1.00	21	carlot-triangle-dissection	hook,dissect,cystic_duct
0.98	234	carlot-triangle-dissection	hook,null_verb,null_target
0.79	359	gallbladder-dissection	grasper,retract,liver
0.79	524	gallbladder-dissection	hook,dissect,gallbladder
0.64	125	gallbladder-packaging	grasper,grasp,specimen_bag
0.63	113	carlot-triangle-dissection	grasper,retract,gallbladder

Observations

Several triplets were almost perfectly associated with a single surgical phase.

Examples include:

* clipper,clip,cystic_duct
* grasper,pack,gallbladder
* hook,dissect,cystic_duct

These triplets appear to represent workflow-defining events rather than generic surgical actions.

In contrast, highly frequent triplets such as:

* grasper,grasp,gallbladder

appeared across multiple phases and therefore showed lower predictive value despite high occurrence counts.

Clinical Interpretation

The appearance of an instrument alone is not always informative.

For example:

irrigator,aspirate,fluid

showed relatively poor phase specificity because irrigation and suction are routinely used throughout laparoscopic cholecystectomy for visualization, smoke evacuation, and minor bleeding control.

This suggests that workflow understanding depends more on task-specific action-target combinations than on instrument presence alone.

Conclusion

Triplets provide substantially stronger workflow information than instruments, verbs, or targets considered separately.

This finding helps explain why modern surgical AI research increasingly focuses on triplet recognition as an intermediate step toward workflow understanding and event recognition.
