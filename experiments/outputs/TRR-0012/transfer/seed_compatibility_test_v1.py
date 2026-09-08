#!/usr/bin/env python3
"""Focused synthetic proof for the TRR0012 approved-seed compatibility map."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.trr0011_capture import apply_fixed_sign_block_update, sha256_file  # noqa: E402

APPROVED_SEEDS = (1101, 1111, 1113)
INHERITED_SEEDS = (1301, 1401)
EXECUTED_SEED = 1101
BLOCK_ELEMENTS = 4096
AMPLITUDE = 1e-2
APPROVED_BY_VARIANT = {
    'layer1_block_rel1e-3': 1101,
    'layer1_block_rel1e-2': 1111,
    'layer3_block_rel1e-3': 1101,
    'layer3_block_rel1e-2': 1111,
    'layer4_null_block_rel1e-2': 1113,
}


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    model.model.layers[1].self_attn = torch.nn.Module()
    model.model.layers[1].self_attn.q_proj = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    base = torch.arange(BLOCK_ELEMENTS, dtype=torch.float32).reshape(64, 64)
    base = (base - base.mean()) / base.std()
    with torch.no_grad():
        model.model.layers[1].self_attn.q_proj.weight.copy_(base.to(torch.bfloat16))
    return model


def _variant(seed: int) -> dict:
    return {
        'variant_id': f'synthetic_seed_{seed}',
        'role': 'artificial_fixed_sign_block',
        'artificial': True,
        'update_location': {
            'layer_index': 1,
            'parameter_path': 'model.layers.1.self_attn.q_proj.weight',
            'block_offset': 0,
            'block_elements': BLOCK_ELEMENTS,
            'relative_parameter_l2': AMPLITUDE,
            'recipe_seed': seed,
            'sign_pattern': 'alternating_seed_phase',
            'block_order': 'flattened_row_major',
        },
    }


def _check_plan(path: Path, *, accept: bool) -> dict:
    plan = json.loads(path.read_text())
    rows = {row['variant_id']: row for row in plan.get('variants', [])}
    errors = []
    for variant_id, approved_seed in APPROVED_BY_VARIANT.items():
        location = rows.get(variant_id, {}).get('update_location')
        if not isinstance(location, dict):
            errors.append(f'{variant_id}: missing update_location')
            continue
        if accept:
            if location.get('approved_recipe_seed') != approved_seed:
                errors.append(f'{variant_id}: approved seed {location.get("approved_recipe_seed")} != {approved_seed}')
            if location.get('executed_recipe_seed') != EXECUTED_SEED:
                errors.append(f'{variant_id}: executed seed annotation changed')
            if location.get('recipe_seed') != EXECUTED_SEED:
                errors.append(f'{variant_id}: runner seed {location.get("recipe_seed")} != {EXECUTED_SEED}')
        else:
            if location.get('approved_recipe_seed') != approved_seed:
                errors.append(f'{variant_id}: approved seed mismatch')
            if location.get('executed_recipe_seed') != EXECUTED_SEED:
                errors.append(f'{variant_id}: executed seed annotation mismatch')
            if location.get('recipe_seed') != EXECUTED_SEED:
                errors.append(f'{variant_id}: runner seed mismatch')
    if accept and errors:
        raise SystemExit('approved executable plan check failed: ' + '; '.join(errors))
    if not accept and not errors:
        raise SystemExit('mismatched plan was not rejected')
    return {
        'path': str(path.resolve()),
        'acceptance_expected': accept,
        'accepted': accept,
        'rejected_mismatch_count': len(errors) if not accept else 0,
        'errors': errors if not accept else [],
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--rejected-plan', type=Path, required=True)
    args = parser.parse_args()
    accepted_plan = _check_plan(args.plan, accept=True)
    rejected_plan = _check_plan(args.rejected_plan, accept=False)
    torch.set_num_threads(1)
    digests = {}
    effective = {}
    values = {}
    for seed in (*APPROVED_SEEDS, *INHERITED_SEEDS, 1100):
        model = _model()
        receipt = apply_fixed_sign_block_update(model, _variant(seed))
        value = model.model.layers[1].self_attn.q_proj.weight.detach().cpu().contiguous()
        digests[str(seed)] = receipt['updated_parameter_digest']
        effective[str(seed)] = receipt['effective_relative_parameter_l2']
        values[str(seed)] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    baseline = values[str(EXECUTED_SEED)]
    odd_equal = all(values[str(seed)] == baseline for seed in (*APPROVED_SEEDS, *INHERITED_SEEDS))
    even_differs = values['1100'] != baseline
    if not odd_equal or not even_differs:
        raise SystemExit('seed parity equivalence failed')
    result = {
        'schema': 'token-reconstruction.trr0012-approved-seed-compatibility-test.v1',
        'status': 'PASS_SYNTHETIC_CONSTRUCTOR_EQUIVALENCE',
        'source_runner': str((ROOT / 'scripts/trr0011_capture.py').resolve()),
        'source_runner_sha256': sha256_file(ROOT / 'scripts/trr0011_capture.py'),
        'constructor': 'scripts.trr0011_capture.apply_fixed_sign_block_update',
        'parameter_path': 'model.layers.1.self_attn.q_proj.weight',
        'block_elements': BLOCK_ELEMENTS,
        'relative_parameter_l2': AMPLITUDE,
        'approved_seeds': list(APPROVED_SEEDS),
        'inherited_recipe_seeds_checked': list(INHERITED_SEEDS),
        'executed_compatibility_seed': EXECUTED_SEED,
        'seed_use': 'Only _fixed_signs(length, seed) consumes recipe_seed; signs use (index + seed) % 2. No RNG or other seed use occurs in the constructor path.',
        'approved_to_executed_mapping': {str(seed): EXECUTED_SEED for seed in APPROVED_SEEDS},
        'inherited_to_executed_mapping_checked': {str(seed): EXECUTED_SEED for seed in INHERITED_SEEDS},
        'updated_parameter_byte_sha256': values,
        'updated_parameter_digest_receipts': digests,
        'effective_relative_l2': effective,
        'odd_seed_updates_exactly_equal': odd_equal,
        'even_control_seed_1100_differs': even_differs,
        'model_loaded': False,
        'gpu_used': False,
        'truth_opened': False,
        'source_selection_performed': False,
        'approved_plan_acceptance': accepted_plan,
        'mismatched_plan_rejection': rejected_plan,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
    if args.receipt.exists() and args.receipt.read_bytes() != data:
        raise SystemExit(f'create-only mismatch: {args.receipt}')
    if not args.receipt.exists():
        args.receipt.write_bytes(data)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
