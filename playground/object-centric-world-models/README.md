# Object-Centric World Models — Playground

Small architecture-level playground, separate from the day-numbered
CholecT50 challenge (`day01_...` onward). The paper discussion itself
(what it argues, why it matters, how it connects to earlier papers) lives
in the Day33 LinkedIn post; this folder is just a small piece of code to
check that understanding by building something concrete on a tiny scale.
It does not need to connect to the day-numbered challenge, though the
code happens to reuse its Day12-14 logic here.

**Paper:** Vakhitov, Ugadiarov, Panov. *Object-Centric World Models Meet
Monte Carlo Tree Search.* arXiv:2601.06604 (2026).

## What's here

`object_graph_demo.py` reuses the Day12/13/14 frame-triplet and state
segmentation logic to render one real moment from VID01 (clipping the
cystic duct) as an Object graph — instrument/target as Nodes, verb as the
Edge between them — across three consecutive states:

```
graph LR
    grasper["grasper"] -- "grasp" --> gallbladder["gallbladder"]
```

is "before clipping"; the `clipper -> cystic_duct` edge appears alongside
it "during clipping", then the graph simplifies again "after clipping".

Run it with:

```
python3 object_graph_demo.py
```

## Not in scope here

No GNN, no learned Dynamics model, no MCTS — just checking that the
paper's Node/Edge framing can be instantiated on data this project
already has. Not a reproduction of the paper.
