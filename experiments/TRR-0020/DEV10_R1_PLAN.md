# Development10 revision1: qualify useful direct-update curvature scale
Dev10 completed80cells and failed to improve the scored initialization. Public-only diagnosis verified gradients share the intended storage and are nonzero; on seed200030 rawbeta.1 improves one update74->88/127 whereasbeta.25 retains74/127. Very smallbeta.0001 performs poorly. This supports a less conservative search range, not a claimed reconstruction result.

Keep the entire declared dev10 method and64iterations. Change only the preregistered beta grid to.003/.01/.03/.1/.25 for raw/white. Tenconfigs*8inputs=80fresh output cells; do not mix previous outputs. Sources/flagsbound and full freeze before current retrospectivelabels.

Add an execution audit perconfiguration: after3repeated graph qualifications, reset and run the same64updates eagerly. Save final/best/per-position tokenoutputs and loss values BEFORE comparing against replay. Require exact tokens and traces; preserve/fail on mismatch. The CPUfactorized quadratic identity and final-vocabulary-entry tests also remain.

Unchanged geometry/resources from qualifieddev10:128positions,64updates, measured~.28s hot inference, expected<3min includingeager tests; childtimeout1200,processreserved<=6GiB,free>=2GiB,admission9000MiB,host>=8GiB,RSS<=10GiB,temp<80C. No canonical claim or active registration.
