# Preserved development failures

Qualification r1 stopped during cold loading before any clip inference: the strict preserved B1 package detected that the native A2 imports already owned its `token_reconstruction` module namespace. No allocator, driver or thermal anomaly occurred. The failed attempt and stderr remain in qualification-r1.

Repair: load the unchanged, fully hash-verified package in a temporary isolated import context, retain its exact loaded B1 class/module objects, verify every package module resolves inside the package code directory, and restore the already-loaded native selector modules afterward. No preserved package bytes or model parameters are changed. Qualification r2 must verify native decisions and repeated predictions before this integration is accepted.
