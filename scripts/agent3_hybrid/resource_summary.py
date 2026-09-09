"""Summarize recorded execution resources after all timed cells have ended."""
import argparse,json
from pathlib import Path
from scripts.agent3_hybrid.common import *

def main(a):
    cells=[]
    for d in DOMAINS:
        for s in STAGES:
            directory=EV/'execution'/f'{d}-{s}';finish=json.loads((directory/'finish.json').read_text());samples=[json.loads(x) for x in (directory/'resources.jsonl').read_text().splitlines()]
            assert finish['status']=='PASS' and not finish['guard_failure']
            samples=[x for x in samples if 'rss_bytes' in x]
            receipt=json.loads((OUT/'predictions'/f'{d}-{s}'/'receipt.json').read_text())
            cells.append({'domain':d,'stage':s,'start_utc':finish['start_utc'],'end_utc':finish['end_utc'],'guarded_wall_seconds':finish['elapsed_seconds'],'worker_wall_seconds':receipt['wall_seconds'],'public_resource_load_seconds':receipt['public_resource_load_seconds'],'serialization_and_final_verification_seconds':receipt['output_io_seconds'],'sample_count':len(samples),'max_sampled_rss_bytes':max(x['rss_bytes'] for x in samples),'min_available_host_bytes':min(x['available_bytes'] for x in samples),'min_free_gpu_mib':min(x['gpu_free_mib'] for x in samples),'max_temperature_c':max(x['temperature_c'] for x in samples),'guard':binding(directory/'finish.json'),'samples':binding(directory/'resources.jsonl')})
    applications=[json.loads(x) for x in (EV/'gpu-activity.jsonl').read_text().splitlines()];maximum=max(len(x['compute_apps']) for x in applications)
    write_json(a.output,{'status':'ALL_EIGHT_GUARDS_PASS','cells':cells,'total_guarded_wall_seconds':sum(x['guarded_wall_seconds'] for x in cells),'gpu_application_samples':len(applications),'max_reported_concurrent_cuda_applications':maximum,'nvml_exclusivity_established':False,'windows_diagnostic':binding(EV/'windows-gpu-engines.json'),'nvml_diagnostic':binding(EV/'process-monitor-diagnostic.json'),'compute_application_pid_set':sorted({line.split(',')[0].strip() for x in applications for line in x['compute_apps']}),'application_log':binding(EV/'gpu-activity.jsonl'),'qualification_retained_separately':True,'timing_scope':'common resident resources and instrumented workloads, not isolated production deployment minima','io_field_scope':'original output_io_seconds includes final implementation hash verification','created_utc':utc()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
