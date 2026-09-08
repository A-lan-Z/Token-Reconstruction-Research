"""Evaluator only: rerender exactly the frozen 32 rows/domain after matrix freeze."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from safetensors.torch import save_file
from scripts.agent3_shortlists.core import binding, verify, write_json
from scripts.agent3_shortlists.evaluate import load_frozen


def main(a):
    started = time.monotonic()
    torch.set_num_threads(2)
    # This revalidates every candidate, prediction, score, input and numerical identity.
    frozen = load_frozen(a.freeze)
    gate_utc = datetime.now(timezone.utc).isoformat()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(a.shared_root).resolve()
    matrix = json.loads(Path(a.input_matrix).read_text())
    captures = [json.loads(verify(c['capture_receipt']).read_text()) for c in matrix['cells']]
    # Bind the exact shared curator implementation before importing any renderer.
    code = captures[0]['code']
    for item in code.values():
        verify(item)
    for item in (root, root / 'src'):
        sys.path.insert(0, str(item))
    capture = importlib.import_module('scripts.trr_p12.capture')
    if Path(capture.__file__).resolve() != verify(code['capture']):
        raise ValueError('shared capture module differs from released implementation')
    expected = json.loads(verify(frozen['source_order']).read_text())['order_first32_by_domain']
    receipt = {
        'schema': 'agent3-evaluator-truth-v1', 'freeze': binding(a.freeze),
        'freeze_revalidated_utc': gate_utc, 'truth_opened_after_complete_freeze': True,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'evaluator_code': binding(__file__), 'shared_code': code,
        'command': sys.argv, 'domains': {}, 'source_labels_printed': False,
        'target_weights_loaded': False, 'p03_accessed': False,
    }
    for domain in ('pile', 'finance'):
        cell = next(c for c in captures if c['observation']['domain'] == domain)
        panel_path = verify(cell['panel'])
        panel, panel_binding, rows = capture.load_panel(panel_path, domain)
        rows = rows[:32]
        ids = [r['record_id'] for r in rows]
        if ids != frozen['record_ids'][domain]:
            raise ValueError('evaluator rows differ from complete freeze')
        source = verify(panel_binding['source_inputs'])
        normalized = capture.p11._normalize_source_inputs(source, root=root, require_tokenizer_dir=True)
        tokenizer = capture.trusted._load_tokenizer(Path(normalized['tokenizer']['path']))
        dataset = capture.trusted._load_arrow_dataset(tuple(Path(x['path']) for x in normalized[domain]['arrow_files']))
        values = []
        for index, declared in enumerate(rows):
            row_index = declared['row_index']
            start, stop = capture.PANEL_RANGES[domain]
            if not start <= row_index < stop:
                raise ValueError('row outside declared source range')
            candidate = capture.trusted._render_row(domain, dataset[row_index], row_index, tokenizer)
            actual = capture.p11._candidate_identity(candidate)
            if any(str(actual.get(k)) != str(declared.get(k)) for k in capture.ROW_FIELDS):
                raise ValueError('source rerender identity changed')
            tokens = torch.tensor(candidate.token_ids[:128], dtype=torch.int64)
            if tokens.shape != (128,) or tokens[0] != 128000:
                raise ValueError('source geometry changed')
            digest = hashlib.sha256(tokens.numpy().astype('<i4').tobytes()).hexdigest()
            if digest != expected[domain][index]['h128_sequence_sha256']:
                raise ValueError('source canonical H128 identity changed')
            values.append(tokens)
        artifact = out / (domain + '.safetensors')
        save_file({'token_ids': torch.stack(values)}, str(artifact))
        receipt['domains'][domain] = {'record_ids': ids, 'artifact': binding(artifact), 'panel': panel_binding}
        del dataset, tokenizer, values
    receipt.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    write_json(out / 'truth-manifest.json', receipt)
    print(json.dumps({'status': 'TRUTH_READY_AFTER_COMPLETE_FREEZE', 'manifest': binding(out / 'truth-manifest.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('freeze', 'shared-root', 'input-matrix', 'output'):
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
