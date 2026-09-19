"""Post-freeze provenance audit; hash verification does not decode source labels."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts/trr0016'))
from support import *

def main():
    receipt=json.loads((X/'prediction_receipt.json').read_text());check_binding(receipt['binding'])
    capture=json.loads((X/'fresh_r3_capture.json').read_text());selection=json.loads((X/'fresh_r3_selection.json').read_text())
    fresh=OUT/'fresh_r3'
    for field,name in [('observations_sha256','observations.safetensors'),('truth_sha256','evaluator_truth.json'),('metadata_sha256','metadata.json')]:
        assert digest(fresh/name)==capture[field]
    assert digest(fresh/'sources.json')==selection['sources_sha256']
    assert digest(X/'PLAN.md')==selection['plan_sha256']
    assert len(receipt['entries'])==1408
    write(X/'provenance_audit.json',{'observations_truth_metadata_source_hashes_match_capture':True,'plan_hash_matches_selection':True,'full_matrix_cells':1408,'implementation_binding_matches':True,'fresh_source_labels_decoded_by_auditor':False,'utc':utc()})
    print('Complete1408-cell receipt and original fresh-capture hashes match',flush=True)
if __name__=='__main__':main()