# TRR-P02 model-free validation receipt

Captured 2026-09-05 before the execution lease. These checks use synthetic
small tensors or static plan/source imports only. No public model was loaded,
no prototype table was opened, no lens was loaded, and no CUDA allocation or
scientific result was produced.

## Exact checks

All commands were run from `/tmp/trr-p02` with the task-local source tree.

```text
python3 -m py_compile src/token_reconstruction/trr_p02/geometry.py scripts/trr_p02/diagnose_geometry.py
# exit 0; no output

PYTHONPATH=src:scripts/trr_p02 pytest -q tests/test_trr_p02_geometry.py
.......                                                                  [100%]
7 passed in 0.83s

PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'scripts/trr_p02')
import diagnose_geometry as d
ids, reference, contexts = d._validate_plan(
    d._read_json(Path('experiments/TRR-P02/plan.json'))
)
assert (len(ids), reference, len(contexts)) == (8, 220, 7)
assert d.SEED == 314159
assert d.RANKING_SCORE_BUFFER_BYTES == 8208384
print('runner plan validation: PASS', len(ids), reference, len(contexts), 'seed', d.SEED)
# output:
# runner plan validation: PASS 8 220 7 seed 314159
PY

PYTHONPATH=src:scripts/trr_p02 python3 - <<'PY'
# Model-free C6 qualification helper and RSS receipt.
import sys
sys.path.insert(0, 'scripts/trr_p02')
import torch
import diagnose_geometry as d
class FakePrefix:
    def forward_full(self, ids):
        batch, tokens = ids.shape
        out = torch.zeros((batch, tokens, d.HIDDEN_SIZE), dtype=torch.bfloat16)
        out[:, :, 0] = ids.to(torch.bfloat16)
        out[:, :, 1] = torch.arange(tokens, dtype=torch.bfloat16)
        return out
check = d.Counters()
result = d._qualify_short_cell(
    FakePrefix(),
    d.ContextSpec('C6_repeat_13_length_3', (d.BOS_TOKEN_ID, 13, 13, 13)),
    d.EXPECTED_CANDIDATE_IDS,
    check,
)
assert result['status'] == 'QUALIFIED_EQUIVALENT'
assert result['endpoint_position'] == 4
assert result['batch_shape'] == [8, d.HIDDEN_SIZE]
checks = []
d._rss_ceiling_check('synthetic', checks)
assert checks[0]['status'] == 'PASS'
print('runner helper smoke: PASS', check.full_calls, check.full_input_tokens)
# output:
# runner helper smoke: PASS 9 80
PY

PYTHONPATH=src:scripts/trr_p02 python3 - <<'PY'
# Model-free full-V score-buffer exercise at Q=4,V=128256.
import torch
from token_reconstruction.trr_p02 import rank_metrics
g = torch.Generator().manual_seed(314159)
prototypes = torch.randn((128256, 4), generator=g)
ids = [127999, 128000, 128001, 128002]
result = rank_metrics(
    prototypes[ids], prototypes, ids, query_chunk_size=4, prototype_chunk_size=8192
)
assert result['top1_ids'].tolist() == ids
assert result['true_rank'].tolist() == [1, 1, 1, 1]
print('full-V synthetic rank buffer: PASS', result['true_rank'].tolist())
# output:
# full-V synthetic rank buffer: PASS [1, 1, 1, 1]
PY
```

The ranking test specifically places the true ID in the last prototype chunk
with tied earlier-block winners; the expected strict rank is 3 and the focused
suite includes local N8-other/self-exclusion, N9 dictionary membership, normal
competitors, and missing-label rejection. The planned numerical command must
use the root-committed implementation SHA, a fresh output directory and
`run_manifest.json`, `CUDA_VISIBLE_DEVICES=''`, eight intra-op threads, one
inter-op thread, `/usr/bin/time -v`, and the separate 8 GiB process RSS ceiling
plus 10 GiB available-host guard.

## Frozen source inventory

- `scripts/trr_p02/diagnose_geometry.py`
- `src/token_reconstruction/trr_p02/geometry.py`
- `src/token_reconstruction/trr_p02/__init__.py`
- `experiments/TRR-P02/plan.json`
- `experiments/TRR-P02/review/design.md`
- `tests/test_trr_p02_geometry.py`

The obsolete untracked `experiments/TRR-P02/plan-v1-draft.json` is excluded.
The setup helper `src/token_reconstruction/trr_p02/exclusion.py` remains a
model-free setup artifact. Root owns the commit, push, PR, aggregate manifest,
and final state update.
