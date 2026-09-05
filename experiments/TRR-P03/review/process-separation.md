# TRR-P03 process-separation note

During today's setup, source task
`01a071d9-7943-7301-9eaa-1bff9e7d4d50` accidentally sent TRR-P03
panel/interface planning to the independent TRR-0004 task. The message
contained no records, source truth, scores, or private data. The receiving task
reported that its methods and fresh records were already frozen and that it
would not use the message for its own work or for the next decision in this
task.

The root corrected routing so P03 scientific messages go only to
`01a07061-9c25-7e13-bcac-12d57c41c666`. P03 uses no scientific output from the
misrouted message or from TRR-0004. Its design inputs remain limited to the
charter, this packet, the published P02 report/plan/code, and the explicitly
named public P01 provenance assets. Setup must preserve this disclosure and
response in the task-local manifest and verify that unrelated-task content
does not enter observation generation, method selection, scoring, or
interpretation.
