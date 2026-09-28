# Development31 capture-lifetime repair
The initial attempt exited1 after all12 T128 cases /36 repetitions, before the
first T40 case. Original source and artifacts are immutable. The receipt preserves
the CUDA stream-capture error and external terminal status.
The probe retained its eager gradient loss variable across contexts; its autograd
AccumulateGrad node belonged to the default stream. Capturing the next length on
the owned stream therefore failed. Delete that reference immediately after backward,
before another graph is captured. No arithmetic, model, batching, input, geometry,
or update constant changes. The new attempt requires exact baseline and saved-output
hash equality for all38completed artifacts from the original attempt. Full logits
still repeat exactly in each attempt. The failed attempt remains excluded from
the completed public matrix; no reconstruction truth was read.
The GPU returned to normal idle memory with no compute processes after exit.
Use a new guarded process, new output directory and receipt; this is a confirmed
terminal failure, not an observation timeout. Original complete-stage timings are
preserved but only the complete r1 matrix is summarized.
