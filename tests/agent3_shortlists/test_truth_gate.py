"""The evaluator must reject an invalid freeze before any source access."""
import json
from types import SimpleNamespace
import pytest
from scripts.agent3_shortlists.materialize_truth import main


def test_invalid_freeze_cannot_materialize_sources(tmp_path):
    frozen = tmp_path / 'invalid-freeze.json'
    frozen.write_text(json.dumps({'schema': 'agent3-complete-matrix-freeze-v1', 'truth_opened': True}))
    destination = tmp_path / 'truth'
    with pytest.raises(ValueError, match='invalid freeze'):
        main(SimpleNamespace(freeze=str(frozen), output=str(destination), shared_root='/nonexistent-shared-source', input_matrix='/nonexistent-matrix'))
    assert not destination.exists()
