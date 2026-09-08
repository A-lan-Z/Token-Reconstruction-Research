"""Export reviewed aggregate-only P11 results without opening truth tensors."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/TRR-P11/private-evaluation/evaluation'
EXPECTED = {
    'score-r1.json': '9f251d0a9fffddc0024c59f61cfaac85dbf83f1de9d0d7931e7fa24a9a1dd26d',
    'accuracy-summary-r1.json': '94a2262f4ae671d255ecfb53d584f17d3544b11280bfa9f455bdabe4246f92e3',
}

def load(name):
    path = BASE / name
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED[name]:
        raise ValueError(f'Changed input: {name}')
    return json.loads(raw), {'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': digest}


def main():
    score, score_ref = load('score-r1.json')
    accuracy, accuracy_ref = load('accuracy-summary-r1.json')
    if score['status'] != 'SCORE_COMPLETE_AFTER_TRUTH' or not score['truth_opened']:
        raise ValueError('Scoring incomplete')
    result = {
        'schema': 'token-reconstruction.trr-p11-public-confirmation-summary.v1',
        'task_id': 'TRR-P11',
        'status': 'CONFIRMATION_SCORED_AFTER_COMPLETE_PREDICTION_FREEZE',
        'source_artifacts': {'score': score_ref, 'accuracy': accuracy_ref, 'freeze': score['freeze']},
        'export_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_calculation': False,
        'raw_truth_or_source_payload_exported': False,
        'primary_panel': {'unique_sources': 512, 'records_per_domain': 256, 'targets_per_source': 2, 'records_per_cell': 256, 'scored_tokens_per_cell': 32512},
        'common_a1_subset': {'selection': 'first128 per domain in frozen order', 'records_per_cell': 128, 'scored_tokens_per_cell': 16256},
        'scored_post_bos_tokens_per_record': 127,
        'cell_order': score['cell_order'],
        'scorer': score['scorer'],
        'cells': {},
        'support_definition': score['support_strata']['definition'],
        'support_reference': {k: score['support_strata'][k] for k in ('status', 'assignment', 'fit_bank_counts')},
        'interpretation': [
            'Expanded-bank B1 gains generalize to unused sources in all four cells of this new paired replication.',
            'Compare B1 with A1+A2 only using the common first128 subset contrasts and paired inventory, not mismatched full-panel denominators.',
            'One paired fit uses fitting seed4010; bootstrap seed9009 and source-selection seed5011 have separate roles.',
            'Original exact-state confirmation remains blocked/incomplete because its selected artifacts are unavailable; original PR20 results remain selection-overlapping development evidence.',
            'No inference was rerun after truth. No canonical all-method comparison or seed-independent reliability is claimed.',
            'Frozen-trace proposal diagnostics and detailed execution costs are supplied in separate additive receipts; opaque trace availability alone is not a computed rank diagnostic.',
        ],
        'p03_holdout_accessed': False,
        'pooling': False,
    }
    for cell in score['cell_order']:
        inventory = score['p10_inventory']['cells'][cell]
        contrasts = score['cells'][cell]
        support = score['support_strata']['cells'][cell]
        result['cells'][cell] = {
            'method_accuracy': accuracy['cells'][cell]['methods'],
            'primary_B1_minus_B0': {k: contrasts['new_B1_minus_new_B0'][k] for k in ('exact_record', 'token_delta')},
            'common128_B1_minus_A1': {k: contrasts['new_B1_minus_a1_a2_first128'][k] for k in ('exact_record', 'token_delta')},
            'paired_inventory': {
                pair: {k: data[k] for k in ('left_method', 'right_method', 'records_exact', 'tokens')}
                for pair, data in inventory['pairwise'].items()
            },
            'support_strata': {k: value for k, value in support.items() if k != 'observation'},
        }
    forbidden = {'record_id', 'record_ids', 'source_text', 'token_ids', 'truth_tokens', 'row_index', 'source_index', 'opaque_id'}
    def validate(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError('Private identity/payload field in export')
            for child in value.values(): validate(child)
        elif isinstance(value, list):
            for child in value: validate(child)
    validate(result)
    path = ROOT / 'experiments/TRR-P11/evaluation/confirmation-summary-r1.json'
    raw = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    with path.open('xb') as handle: handle.write(raw)
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'cells': len(result['cells']), 'status': result['status']}))

if __name__ == '__main__':
    main()
