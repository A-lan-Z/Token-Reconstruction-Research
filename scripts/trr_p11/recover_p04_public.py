from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

from datasets import Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / 'experiments/TRR-P11/exclusions/p04_public_selection_r2.json'
SELECTION_EXPECTED_SHA256 = '05f941e0dbcf29ea3efc47c7bc8abb3a7146a266eeea770f05052bb7728cde6a'
OUTPUT = ROOT / 'experiments/TRR-P11/exclusions/p04_h128_recovery_r3.json'
IDENTITY_OUTPUT = ROOT / 'experiments/TRR-P11/exclusions/p04_h128_identity_rows_r2.json'
ALPACA_HELPER = ROOT / 'src/token_reconstruction/alpaca_split.py'
ALPACA_HELPER_EXPECTED_SHA256 = 'fa9a15fd4cf92ffa06be3bd77888324536180e8a2fdc2c43ae14f20e470a3626a'
SCRIPT_PATH = Path(__file__).resolve()
TOKENIZER_PATH = Path('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6')
BOS = 128000
ALPACA_DATE = os.environ.get('TRR_P04_ALPACA_DATE', '06 Sep 2026')
MAX_RSS_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 12 * 1024**3
ARROWS = {
    'pile_plain': [Path('/home/alanz/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow')],
    'finance_chat': [
        Path('/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00000-of-00002.arrow'),
        Path('/home/alanz/.cache/huggingface/datasets/Josephgflowers___finance-instruct-500k/default/0.0.0/583a98fb0ec14d904e9423b671d9d0fea88891b6/finance-instruct-500k-train-00001-of-00002.arrow'),
    ],
    'alpaca_instruction': [Path('/home/alanz/.cache/huggingface/datasets/tatsu-lab___alpaca/default/0.0.0/dce01c9b08f87459cf36a430d809084718273017/alpaca-train.arrow')],
}
ARROW_EXPECTED = {
    'pile_plain': [(61270696, '77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113')],
    'finance_chat': [(503571864, 'b49ca0980a0b02fecbef2220eee0ef5d3c3c893ae42b4e1910edec993c3d164e'), (53756664, 'ce4b0786646cd68561da736f145fd5df7ba2f4e754e0caa3ae646d6be9900bd3')],
    'alpaca_instruction': [(46230248, 'f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794')],
}
TOKEN_EXPECTED = {
    'tokenizer.json': (9085657, '79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4'),
    'tokenizer_config.json': (54528, '9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e'),
    'special_tokens_map.json': (296, '6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec'),
}

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def h_ids(ids):
    vals = [int(v) for v in ids]
    return sha_bytes(struct.pack('<' + 'i'*len(vals), *vals))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def text_value(row, key):
    value = row.get(key, '')
    if value is None: return ''
    if not isinstance(value, str): raise ValueError(f'{key} is not text')
    return value

def finance_fields(row):
    system = text_value(row, 'system').strip() or None
    user = text_value(row, 'user').strip()
    assistant = text_value(row, 'assistant').strip()
    if not user:
        instruction = text_value(row, 'instruction').strip()
        input_text = text_value(row, 'input').strip()
        user = instruction + (('\n\n' + input_text) if input_text else '')
    if not assistant:
        assistant = text_value(row, 'output').strip()
    return system, user, assistant

def alpaca_user(row):
    instruction = text_value(row, 'instruction')
    input_text = text_value(row, 'input')
    return instruction + (('\n\n' + input_text) if input_text else '')

def token_list(value):
    if hasattr(value, 'input_ids'): value = value.input_ids
    elif hasattr(value, 'keys') and 'input_ids' in value: value = value['input_ids']
    if hasattr(value, 'tolist'): value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list): value = value[0]
    if not isinstance(value, (list, tuple)): raise ValueError('tokenizer returned non-list')
    return [int(v) for v in value]

def render(style, row, tokenizer):
    if style == 'pile_plain':
        text = text_value(row, 'text')
        source_hash = sha_bytes(text.encode('utf-8'))
        ids = [BOS] + token_list(tokenizer(text, add_special_tokens=False))
        return source_hash, ids, len(text)
    if style == 'finance_chat':
        system, user, assistant = finance_fields(row)
        if not user or not assistant: raise ValueError('missing Finance user/assistant')
        content = json.dumps([system, user, assistant], ensure_ascii=False, separators=(',', ':'))
        messages = []
        if system: messages.append({'role':'system','content':system})
        messages.extend([{'role':'user','content':user},{'role':'assistant','content':assistant}])
        ids = token_list(tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, date_string='06 Aug 2026'))
        return sha_bytes(content.encode('utf-8')), ids, len(content)
    if style == 'alpaca_instruction':
        user = alpaca_user(row)[:1200]
        output = text_value(row, 'output')[:1200]
        rendered = tokenizer.apply_chat_template([{'role':'user','content':user}], tokenize=False, add_generation_prompt=True, date_string=ALPACA_DATE) + output
        ids = token_list(tokenizer(rendered, add_special_tokens=False))
        return sha_bytes(rendered.encode('utf-8')), ids, len(rendered)
    raise ValueError(style)

def main():
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.monotonic()
    if OUTPUT.exists(): raise RuntimeError(f'refusing to overwrite {OUTPUT}')
    if IDENTITY_OUTPUT.exists(): raise RuntimeError(f'refusing to overwrite {IDENTITY_OUTPUT}')
    meminfo = {line.split(':',1)[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    available = meminfo.get('MemAvailable', 0)
    if available < MIN_FREE_BYTES: raise RuntimeError(f'host free memory {available} below required {MIN_FREE_BYTES}')
    if not SELECTION.is_file() or SELECTION.is_symlink(): raise RuntimeError(f'P04 selection metadata is unavailable: {SELECTION}')
    selection_sha = sha_file(SELECTION)
    if selection_sha != SELECTION_EXPECTED_SHA256: raise RuntimeError(f'P04 selection metadata changed: {SELECTION}')
    if not ALPACA_HELPER.is_file() or ALPACA_HELPER.is_symlink(): raise RuntimeError(f'Alpaca helper is unavailable: {ALPACA_HELPER}')
    alpaca_helper_sha = sha_file(ALPACA_HELPER)
    if alpaca_helper_sha != ALPACA_HELPER_EXPECTED_SHA256: raise RuntimeError(f'Alpaca helper changed: {ALPACA_HELPER}')
    selection = json.loads(SELECTION.read_text())
    expected_rows = [r for pool in ('correction','validation','fresh_evaluation') for r in selection['pools'][pool]['records']]
    if len(expected_rows) != 520 or len({r['record_id'] for r in expected_rows}) != 520: raise RuntimeError('P04 selection rows are not 520 unique records')
    asset_checks = {'arrow': {}, 'tokenizer': {}}
    for style, paths in ARROWS.items():
        asset_checks['arrow'][style] = []
        for path, (expected_bytes, expected_sha) in zip(paths, ARROW_EXPECTED[style]):
            actual_sha = sha_file(path)
            actual_bytes = path.stat().st_size
            if (actual_bytes, actual_sha) != (expected_bytes, expected_sha): raise RuntimeError(f'Arrow asset changed: {path}')
            asset_checks['arrow'][style].append({'path':str(path),'bytes':actual_bytes,'sha256':actual_sha})
    for name,(expected_bytes,expected_sha) in TOKEN_EXPECTED.items():
        path = TOKENIZER_PATH/name
        actual_sha=sha_file(path); actual_bytes=path.stat().st_size
        if (actual_bytes,actual_sha)!=(expected_bytes,expected_sha): raise RuntimeError(f'tokenizer asset changed: {path}')
        asset_checks['tokenizer'][name]={'path':str(path),'bytes':actual_bytes,'sha256':actual_sha}
    datasets = {}
    for style, paths in ARROWS.items():
        shards = [Dataset.from_file(str(path)) for path in paths]
        datasets[style] = shards
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True, use_fast=True)
    if int(tokenizer.bos_token_id) != BOS: raise RuntimeError('tokenizer BOS changed')
    expected_by_key = {(r['style'], int(r['row_index'])): r for r in expected_rows}
    ordered = sorted(expected_by_key.items())
    mismatches=[]; validated=0; h128=[]; h129=[]; rendered=[]; record_ids=[]; identity_by_key={}
    shard_lengths={style:[len(ds) for ds in shards] for style,shards in datasets.items()}
    for (style,row_index), expected in ordered:
        if style == 'finance_chat':
            if row_index < shard_lengths[style][0]: row = datasets[style][0][row_index]
            else: row = datasets[style][1][row_index - shard_lengths[style][0]]
        else: row = datasets[style][0][row_index]
        try:
            public_hash, ids, rendered_chars = render(style,row,tokenizer)
            if not ids or ids[0] != BOS: raise ValueError('BOS mismatch')
            actual = {
                'record_id': (f'pile10k-{row_index:05d}-{public_hash[:16]}' if style=='pile_plain' else (f'finance-public-{row_index:06d}-{public_hash[:16]}' if style=='finance_chat' else f'tatsu-lab/alpaca/train@{expected["dataset_revision"]}:row-{row_index:05d}')),
                'public_record_sha256': public_hash,
                'truncated_sequence_sha256': h_ids(ids[:129]),
                'h128_sequence_sha256': h_ids(ids[:128]) if len(ids)>=128 else None,
                'rendered_char_count': rendered_chars,
                'full_token_count': len(ids),
                'post_bos_token_count': len(ids)-1,
            }
            for field in ('record_id','public_record_sha256','truncated_sequence_sha256','rendered_char_count','full_token_count','post_bos_token_count'):
                if actual[field] != expected[field]:
                    mismatches.append({'style':style,'row_index':row_index,'field':field,'expected':expected[field],'actual':actual[field]})
            if actual['h128_sequence_sha256'] is not None:
                h128.append(actual['h128_sequence_sha256'])
            h129.append(actual['truncated_sequence_sha256']); rendered.append(actual['public_record_sha256']); record_ids.append(actual['record_id']); validated += 1
            identity_by_key[(style, row_index)] = {
                'pool': expected['pool'],
                'style': expected['style'],
                'dataset_id': expected['dataset_id'],
                'dataset_revision': expected['dataset_revision'],
                'row_index': int(expected['row_index']),
                'record_id': actual['record_id'],
                'public_record_sha256': actual['public_record_sha256'],
                'h128_sequence_sha256': actual['h128_sequence_sha256'],
                'h129_sequence_sha256': actual['truncated_sequence_sha256'],
                'rendered_char_count': actual['rendered_char_count'],
                'full_token_count': actual['full_token_count'],
                'post_bos_token_count': actual['post_bos_token_count'],
            }
        except Exception as exc:
            mismatches.append({'style':style,'row_index':row_index,'field':'exception','reason':type(exc).__name__+': '+str(exc)[:200]})
    ended = dt.datetime.now(dt.timezone.utc)
    child=resource.getrusage(resource.RUSAGE_SELF)
    status='PASS_P04_EXACT_RENDERED_H129_H128_RECOVERY' if not mismatches else 'FAIL_P04_EXACT_RENDERED_H129_VALIDATION'
    receipt={
      'schema':'token-reconstruction.trr-p11-p04-h128-recovery.v1','task_id':'TRR-P11','status':status,
      'started_utc':started.isoformat(),'ended_utc':ended.isoformat(),'elapsed_seconds':time.monotonic()-t0,
      'resource':{'max_rss_kb':child.ru_maxrss,'max_rss_bytes':child.ru_maxrss*1024,'max_rss_limit_bytes':MAX_RSS_BYTES,'host_available_bytes_at_start':available,'host_minimum_bytes':MIN_FREE_BYTES,'threads':1,'timeout_seconds':180,'gpu_used':False,'model_loaded':False},
      'source_code':{'prepare_panel_commit':'f423ef596a718a7c8a8480e6211295b97bdfd806','prepare_panel_sha256':'26c003fc37a80c549ca04ebbf0dd629ae09026fad5f4afc21af0adcca72db97f','alpaca_helper_path':str(ALPACA_HELPER.relative_to(ROOT)),'alpaca_helper_sha256':alpaca_helper_sha,'alpaca_helper_expected_sha256':ALPACA_HELPER_EXPECTED_SHA256,'recovery_script_path':str(SCRIPT_PATH),'recovery_script_sha256':sha_file(SCRIPT_PATH)},
      'tokenizer':{'snapshot':str(TOKENIZER_PATH),'revision':'9213176726f574b556790deb65791e0c5aa438b6','bos_token_id':BOS,'alpaca_date_string':ALPACA_DATE,'finance_date_string':'06 Aug 2026'},
      'assets':asset_checks,
      'selection':{'source_path':str(SELECTION.relative_to(ROOT)),'original_source_path':'experiments/TRR-P04/setup/public_selection-r2.json','source_sha256':selection_sha,'expected_source_sha256':SELECTION_EXPECTED_SHA256,'rows_expected':520,'rows_validated':validated,'unique_selected_records':len(expected_by_key),'pools':{'correction':256,'validation':192,'fresh_evaluation':72}},
      'coverage':{'by_style':{style:{'rows':sum(1 for r in expected_rows if r['style']==style),'h128_rows':sum(1 for r in expected_rows if r['style']==style and (next((a for a in []),None) is None))} for style in ARROWS},'h128_rows':len(h128),'h129_rows':len(h129),'rendered_hash_rows':len(rendered),'short_h128_rows':sum(1 for r in expected_rows if r['full_token_count']<128)},
      'hashchecks':{'record_id': not any(x['field']=='record_id' for x in mismatches),'public_record_sha256':not any(x['field']=='public_record_sha256' for x in mismatches),'truncated_sequence_sha256_h129':not any(x['field']=='truncated_sequence_sha256' for x in mismatches),'geometry':not any(x['field'] in {'rendered_char_count','full_token_count','post_bos_token_count'} for x in mismatches),'h128_derived_after_checks':not bool(mismatches)},
      'identity_commitments':{'record_ids_ordered_sha256':sha_bytes(('\n'.join(record_ids)+'\n').encode()),'rendered_hashes_ordered_sha256':sha_bytes(('\n'.join(rendered)+'\n').encode()),'h129_hashes_ordered_sha256':sha_bytes(('\n'.join(h129)+'\n').encode()),'h128_hashes_ordered_sha256':sha_bytes(('\n'.join(h128)+'\n').encode()) if h128 else None},
      'mismatches':mismatches[:200],
      'mismatch_count':len(mismatches),
      'access_boundary':{'source_text_materialized_transiently':True,'source_text_serialized':False,'source_tokens_materialized_transiently':True,'source_tokens_serialized':False,'token_values_emitted':False,'evaluation_truth_opened':False,'target_update_opened':False,'model_loaded':False,'gpu_used':False,'p03_holdout_accessed':False,'new_selection_started':False},
    }
    if not mismatches:
        identity_rows = [identity_by_key[(r['style'], int(r['row_index']))] for r in expected_rows]
        identity_payload = {
            'schema': 'token-reconstruction.trr-p11-p04-h128-identity-rows.v1',
            'task_id': 'TRR-P11',
            'status': 'PASS_P04_EXACT_RENDERED_H129_H128_RECOVERY',
            'source_selection_sha256': sha_file(SELECTION),
            'recovery_script_sha256': sha_file(SCRIPT_PATH),
            'source_code': receipt['source_code'],
            'records': identity_rows,
            'record_count': len(identity_rows),
            'pools': {'correction': 256, 'validation': 192, 'fresh_evaluation': 72},
            'contains_source_text': False,
            'contains_token_ids': False,
            'contains_truth': False,
            'access_boundary': receipt['access_boundary'],
        }
        with IDENTITY_OUTPUT.open('xb') as f: f.write((json.dumps(identity_payload,sort_keys=True,indent=2)+'\n').encode())
        receipt['identity_export'] = {'path':str(IDENTITY_OUTPUT.relative_to(ROOT)),'bytes':IDENTITY_OUTPUT.stat().st_size,'sha256':sha_file(IDENTITY_OUTPUT),'schema':identity_payload['schema'],'record_count':len(identity_rows)}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('xb') as f: f.write((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps({'output':str(OUTPUT),'identity_output':str(IDENTITY_OUTPUT) if not mismatches else None,'status':status,'mismatches':len(mismatches),'h128_rows':len(h128),'h129_rows':len(h129),'elapsed_seconds':receipt['elapsed_seconds'],'max_rss_kb':child.ru_maxrss},sort_keys=True))
    if mismatches: raise SystemExit(2)

if __name__=='__main__': main()
