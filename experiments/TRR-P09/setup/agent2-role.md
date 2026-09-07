# TRR-P09 Agent 2 role

Agent 2 owns public fitting-bank preparation and fixed-readout controls for the shared P09 study. This includes nested public-bank definitions, public metadata/tokenization and exclusion audits, fit/dev partition bookkeeping, activation-pair preparation after the common contract is frozen, fixed-readout control wiring, and preparation/cost provenance. Any future fitting or public forward requires the agreed contract, resource preflight, and root authorization.

Agent 1 owns the directional-readout implementation and fits, the shared comparison contract, final evaluation execution, truth-scoring interface, and final scientific evaluation. Agent 2 does not edit Agent 1's worktree or directional method, choose final evaluation settings from answers, or open fresh evaluation truth.

The unchanged published TRR-0009 state is the fixed-readout starting reference. P09 keeps A2 out of deployed inference and does not add contextual access, staging, teacher ranking, routing, or ensembles. Public fitting material remains separate from final evaluation material; source text, source token IDs, target labels, and truth payloads stay outside student/control metadata until the declared post-freeze scoring boundary.

This setup commit records inventory and resource evidence only. No P09 source generation, model load, fit, forward, evaluation selection, or truth access occurred.
