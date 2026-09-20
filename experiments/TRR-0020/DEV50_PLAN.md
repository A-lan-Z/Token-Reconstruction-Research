# Development50: complete reconstruction with moment-bound updates

Exploratory comparison; no new active canonical method. Development49R1
passed84independent CPUcases and58GPUcells/174repetitions, including eight
original public states. Scalar roots agree with the torch FP64 reference after
FP32 rate casting, with zero observed rate and output-probability differences.
All original control anchors and all1440graph replay checks pass.
The isolated update is about3.1-3.3xfaster at128positions and2.1-2.6xat40.
This does not establish decoder speed or reconstruction quality.

Freeze96cells: three rules (unchanged cached-probability control, Bennett bound,
two-point bound) xactual16/32/64/128updates xeight existing development inputs.
Three repetitions per cell; no candidate list, verifier, trained predictor,
geometry changes, padding or microbatching. All128256tokens remain eligible.
Moment constants and arithmetic remain exactly as qualified in development49R1.
Both new variants use zero full-vocabulary KL root evaluations;48FP64scalar
bisections are performed per position. Trace columns identify upper bounds,
not measured actual KL. Do not mislabel them as the old exact-KL diagnostics.

For each isolated rule, qualify both native geometries and all four stopping
budgets with3replays and1eager decode. Check exact repeats and eager/replay
outputs/traces. Control must reproduce the frozen development48R1control
at16/32/64 and development46reuse at128. New rules preserve warm initialization
only. Independent embedding-cotangent gradient checks pass at both geometries.
For each new rule, eager128-step qualification additionally measures direct
FP64KL on CPU at updates1/9/33/65/128, requiring actual<=budget+2e-5 and>=-2e-5.
These diagnostic computations are excluded from timed reconstruction repeats.
Ten per-rule trajectory checks are required. Native128positions128steps goes
first; release the rest only after its qualification under the guard.

Freeze the entire96cellmatrix before opening retrospective truth. Require all
three negative truth gates. Keep last-iterate, best-objective and best-position-
error outputs, exact-record counts and all timing phases. Development inputs
were scored earlier and cannot constitute fresh confirmation. A selected
contender must be registered and evaluated in both canonical setups with the
complete active-method matrix before a comparable replacement claim.

Per-record times include initialization, updates, diagnostics, synchronization
and output transfer; preparation, graph capture and artifact I/O are separate.
Workers run two_point, Bennett, control sequentially, without randomized order.
Current paired controls are required; do not compare absolute times across runs.

Preflight: complete prior workers peak5.104GiB, moment-update qualification3.873GiB.
Conservative complete-worker estimate7GiB, initialfree>=10GiB, no competing
compute. Outer runtime guard requires>=2GiBfree,host>=8GiB,RSS<=10GiB,temp<80C;
inner guard>=3GiBfree/reserved<=8GiB. CPUreference arrays max~650MBtemporary.
Estimated<=500s including public qualification and288reconstruction repeats;
guard timeout1200s. Preserve all failed or excluded attempts independently.
