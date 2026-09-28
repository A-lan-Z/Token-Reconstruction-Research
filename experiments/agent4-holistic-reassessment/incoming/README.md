# Agent 4 holistic reassessment

This package contains a research synthesis, a bounded decision brief, and runnable analytical checks.

- `RESEARCH_REPORT.md`: findings, limits, hypotheses, decision criteria and primary sources.
- `AGENT4_DECISION_BRIEF.md`: evidence-gated follow-up for the owner of the local rescue implementation.
- `analytical_checks.py`: calculations and deliberately constructed counterexamples using NumPy.
- `analytical_results.json`: recorded output of that script.

Run `python analytical_checks.py` to regenerate the JSON. No network access, trained model, GPU or private source data is required. These checks are not new Llama reconstruction results and do not demonstrate a rescued implementation.

The most recent empirical numbers were supplied in the conversation. Local commit `ce04a8d`, its completed full report and its per-position traces were not accessible to this reviewer. The report preserves that limitation; it does not infer local implementation details from upstream SIPIT or from the earlier rescue proposal.
