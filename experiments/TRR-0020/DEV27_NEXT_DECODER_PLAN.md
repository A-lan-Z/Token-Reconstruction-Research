# Next development27 decoder experiment

The current-token forward/J/JT are qualified against independent actual-prefix autograd.
The cheaper public comparison completed24cells/72executions, with six exact current-forward
anchors and eighteen exact CGLS1 directions. polyak0.25 uses one transpose product at roughly3ms;
cg1 uses one transpose plus one JVP at about5.5-6.2ms. The more aggressive polyak1 full step
increased nonlinear mismatch at all six public contexts and is not selected.

Implement an end-to-end committed-context decoder using these two fixed cheap rules. Start
from the unchanged full-vocabulary Gini initializer at warm0 or warm64, then solve each unknown
position using only already committed recovered tokens as past context. Use one or four current
continuous updates, with the qualified public median embedding-norm cap. Final readout is the
existing white-cosine full-vocabulary rule. Commit that final token solely to construct subsequent
K/V; never use its commit activation to reject, rerank or replace it. No proposed K-list exists.

Prospective family: warm0/64 x polyak0.25/cg1 x current updates1/4 = eight configurations.
Use the same eight retrospective development records, giving64cells. Implementation and exact
source/asset bindings must be complete before this grid starts. The current prototype is uncaptured;
first measure quality honestly, then qualify any execution optimization separately. Do not claim
the direction-only3ms as end-to-end speed.

Requirements before reconstruction:
- preserve qualified single-position, cheap-direction and stable-CGLS sources;
- compare whole-decoder repeat outputs on public128/40 fixtures, including the largest warm64/cg1/4
  case and full-vocabulary readout formula references;
- require exact warm trajectory and initial-mixture anchors to existing development evidence;
- account separately for warm inference, every current forward and derivative product, full-vocabulary
  readout, all committed-token forwards, model/table setup, synchronization, transfer and I/O;
- calculate live memory/runtime preflight from the measured public primitive costs; one GPU job only;
- freeze all64 outputs before retrospective scoring, test negative truth gates and archive all raw outputs;
- keep the active68-cell canonical registry unchanged unless a fixed rule is selected, then evaluate
  both canonical setups and the complete active matrix.

All128256 vocabulary entries remain eligible at final readout. This uses a prefix-derived table for
initialization/readout, not a fitted A1 or a candidate proposer. It is not lookup-free. A future fixed-size
cache, CUDA graph, precision change or batching adaptation needs its own declared numerical identity
and appropriate equivalence checks; do not silently substitute it for the initial decoder.
