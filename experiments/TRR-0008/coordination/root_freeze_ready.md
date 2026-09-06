# TRR-0008 root-freeze handoff

The selector and evaluation adapters are source-stable for root review. The decision contract is frozen at `2026-09-06T13:19:44Z`. The current task-local state has no selection, capture, prediction, reservation, or truth output. The producer-authored P06 receipt is bound in planning status: `experiments/TRR-0008/planning/approved_opaque/p06_hash_construction_receipt.json`, 2,339 bytes, SHA-256 `b06ca9ccae7b831318604351ce76f183a8c2745780494e20b263279f146ba92c`.

Audit of the earlier approval block: the rejected action attempted to create a task-local P06 confirmation receipt from parent-coordination text and mark the source-hash gate verified. Automatic review rejected it because it would persistently elevate an untrusted coordination claim without a trusted user authorization. No workaround or retry was used. The producer-authored receipt above is the approved evidence that resolved that block.

The remaining root-only action is owner authorization for the create-only selection command. Preserve the exact producer rules in the bound receipt: Pile hashes original source text UTF-8 bytes without stripping or chat rendering; Finance hashes compact UTF-8 canonical JSON `[system,user,assistant]` with `sort_keys=True`, `separators=(',', ':')`, and `ensure_ascii=False`, after the documented field fallback and stripping rules. Keep the H128 sequence rule as SHA-256 over exactly 128 little-endian signed int32 IDs including BOS. Keep `selection_performed=false` and `truth_created_or_opened=false` until the create-only command completes.

After those edits, root may run the following command from the TRR-0008 worktree. This command is recorded for review only and was not executed here:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python3 scripts/trr0008_select_public.py select \
  --repository-root . \
  --decision-contract experiments/TRR-0008/planning/decision_contract.json \
  --planning-status experiments/TRR-0008/coordination/planning_status.json \
  --inventory experiments/TRR-0008/planning/identity_inventory_1thread.json \
  --method-freeze experiments/TRR-0007/method_freeze.json \
  --tokenizer /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --pile-arrow /home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow \
  --finance-arrow /home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow \
  /home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow \
  --output experiments/TRR-0008/selection/source_selection.json \
  --exclusions-output experiments/TRR-0008/selection/source_exclusions.json
```

The command remains intentionally unrun in this task. It will revalidate the owner-frozen contract, method freeze, inventory, timing receipt, source descriptors, approved identity exclusions, and source-hash convention before opening public Arrow rows.
