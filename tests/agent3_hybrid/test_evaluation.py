import numpy as np
import pytest
from scripts.agent3_hybrid.evaluate import error_counts,contrast,freeze

def test_inclusion_and_prior_error_are_separate():
    truth=np.array([[128000,1,2,3,4],[128000,1,2,3,4]])
    predictions=np.array([[128000,1,9,8,4],[128000,9,2,9,4]])
    candidates=np.array([[[1,7],[9,2],[8,7],[4,7]],[[9,8],[2,7],[9,3],[4,7]]])
    x,correct,included=error_counts(predictions,candidates,truth)
    assert x['wrong_despite_inclusion']==2
    assert x['wrong_with_omission']==2
    assert x['wrong_after_earlier_wrong_commitment']==2
    assert x['included_but_wrong_after_earlier_wrong']==1
    assert x['omitted_and_wrong_after_earlier_wrong']==1
    assert x['first_error_positions']==[2,1]
    assert x['correct_after_earlier_wrong']==3
    assert x['exact_clips']==0

def test_quality_and_runtime_are_separate_criteria():
    correct=np.ones((32,127),dtype=bool);worse=correct.copy();worse[0,0]=False
    x=contrast(worse,correct,np.ones(32),np.full(32,2.))
    assert not x['empirical_quality_limits_met'] # one lost clip exceeds2pp
    assert x['empirical_acceleration_met']
    identical=contrast(correct,correct,np.ones(32),np.ones(32))
    assert identical['empirical_quality_limits_met'] and not identical['empirical_acceleration_met']
    assert identical['record_loss_probability_one_sided95_upper']>.08
    assert identical['equivalence_established'] is False
