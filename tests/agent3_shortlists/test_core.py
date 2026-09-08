import numpy as np
import pytest
import torch
from scripts.agent3_shortlists.core import rank_scores, rank_truth, metrics, paired
from scripts.agent3_shortlists.evaluate import freeze, load_frozen


def test_ties_across_cutoff_and_nested_lists():
    s=torch.tensor([[0.,1.,1.,1.,-1.],[9.,2.,3.,4.,5.]])
    ids,scores=rank_scores(s,3)
    assert ids.tolist()==[[1,2,3],[0,4,3]]
    for k in (1,2,3):
        assert torch.equal(rank_scores(s,k)[0],ids[:,:k])


def test_nonfinite_and_duplicate_rejected():
    with pytest.raises(ValueError):rank_scores(torch.tensor([[float('nan'),0.]]),1)
    with pytest.raises(ValueError):rank_truth(torch.tensor([[[2,2]]]),torch.tensor([[2]]))


def test_true_rank_and_conditional_and_exact_clip():
    candidates=torch.tensor([[[3,4,5],[6,7,8]],[[1,2,3],[7,8,9]]])
    truth=torch.tensor([[4,6],[4,9]])
    ranks=rank_truth(candidates,truth)
    assert ranks.tolist()==[[2,1],[4,3]]
    m=metrics(ranks,2)
    assert m['included_tokens']==2 and m['complete_clips']==1
    assert m['recall_among_top1_wrong']==1/3
    assert m['per_record_omissions']==[0,2]


def test_zero_denominator_and_no_fake_equivalence():
    m=metrics(np.ones((2,3),dtype=int),1)
    assert m['recall_among_top1_wrong'] is None
    assert m['conditional_undefined_bootstrap_draws']==10000
    p=paired(np.ones((32,127)),np.ones((32,127)))
    assert p['record_loss_probability_one_sided_95_upper']>.08
    assert not p['equivalence_established']


def test_freeze_rejects_partial_before_any_truth(tmp_path):
    with pytest.raises(ValueError,match='eight'):freeze([],tmp_path/'freeze.json')


def test_complete_freeze_then_score_and_tamper(tmp_path):
    import json
    from safetensors.torch import save_file
    from scripts.agent3_shortlists.core import binding,write_json
    from scripts.agent3_shortlists.evaluate import score
    receipt_paths=[]
    ids={d:[f'{d}/{i}' for i in range(32)] for d in ('pile','finance')}
    obs=tmp_path/'obs.bin';obs.write_bytes(b'synthetic observation placeholder')
    candidates=torch.arange(256,dtype=torch.int64).expand(32,127,-1).clone()
    values=(-torch.arange(256,dtype=torch.float32)).expand(32,127,-1).clone()
    predictions=torch.zeros((32,128),dtype=torch.int64);predictions[:,0]=128000
    pred=tmp_path/'pred.safetensors'
    save_file({'candidates':candidates,'scores':values,'predictions':predictions},pred)
    for d in ids:
        for s in (0,64,128,256):
            contract=tmp_path/f'{d}-{s}-contract.json'
            write_json(contract,{'domain':d,'stage':s,'record_ids':ids[d],'observations':binding(obs)})
            receipt=tmp_path/f'{d}-{s}.json'
            write_json(receipt,{'domain':d,'stage':s,'record_ids':ids[d],'status':'FROZEN_NO_TRUTH','contract':binding(contract),
                               'truth_opened':False,'target_weights_loaded':False,
                               'code_commit':'synthetic','code_files':[binding(obs)],'state_sha256':'synthetic','readout_sha256':'synthetic','package_files_sha256':'synthetic','environment':{'fixture':True},'package_manifest':binding(obs),'lens':binding(obs),'reference':binding(obs),
                               'methods':[{'method':m,'artifact':binding(pred)} for m in ('a1','b1')]})
            receipt_paths.append(receipt)
    f=tmp_path/'freeze.json';freeze(receipt_paths,f)
    assert not (tmp_path/'truth.json').exists()
    # Only now construct a synthetic evaluator truth artifact.
    labels=predictions.clone();labels[:,1:]=3
    truth=tmp_path/'truth.safetensors';save_file({'token_ids':labels},truth)
    manifest=tmp_path/'truth.json';write_json(manifest,{'domains':{d:{'record_ids':ids[d],'artifact':binding(truth)} for d in ids}})
    result=score(f,manifest,tmp_path/'result.json')
    assert result['globally_empirically_promising_budgets']==[8,16,32]
    assert result['cells'][0]['budgets'][0]['omitted_tokens']==32*127
    obs.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='binding changed'):load_frozen(f)
