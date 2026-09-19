from token_reconstruction.prefix_fragments import expand_candidates, suffix_ids, build_suffix_cache
import pytest

class ToyTokenizer:
    texts = [" surprised", "ised", "ed", "d", "\ufffd", "x"]
    def decode(self, ids, **kwargs):return self.texts[ids[0]]
    def batch_decode(self, ids, **kwargs):return [self.decode(x) for x in ids]
    def encode(self, s, **kwargs):
        if s == "ised":return [1]
        if s == "ed":return [2]
        if s == "d":return [3]
        return [5]
    def __call__(self, texts, **kwargs):return {"input_ids":[self.encode(x) for x in texts]}

def test_whole_word_produces_its_missing_suffix_piece():
    t=ToyTokenizer();cache=build_suffix_cache(t,6,chunk=2)
    assert 1 in cache[0]
    assert cache==[suffix_ids(t,v) for v in range(6)]
    assert cache[4]==()

def test_round_robin_preserves_base_and_deduplicates():
    cache=[(3,4),(4,5),(),(),(),()]
    assert expand_candidates([0,1,0],cache,7)==[0,1,3,4,5,0,0]

def test_budget_and_order_are_explicit():
    assert expand_candidates([0,1],[(2,3),(4,5),(),(),(),()],4)==[0,1,2,4]
    with pytest.raises(ValueError):expand_candidates([],[],512)
    with pytest.raises(ValueError):expand_candidates([0,1],[(),()],1)
