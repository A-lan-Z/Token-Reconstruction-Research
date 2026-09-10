# Post-result cost control

The second cohort shows 360/384 joint versus 340/384 narrow sequential, but
joint consumes more total transformed positions and vocabulary scoring.
Before recommending a Llama run, compare an ordinary sequential solver with
16 proposals/step. This single setting is selected as a transparent wider-search
control (not tuned for correctness). Keep the four-round cap and all other
settings identical. Evaluate on the already opened second cohort and clearly
label this comparison retrospective. A wider simple search that outperforms the
joint design at lower measured cost defeats the practical recommendation even
if joint evidence is useful at matched narrow proposal width.
