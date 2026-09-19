from token_reconstruction.prefix_fragment_mixed import fragment_pool,select_fragments

def test_votes_count_distinct_parents_and_never_include_base():
    cache=[(2,3),(3,4),(),(),()]
    base,pool,votes=fragment_pool([0,0,1],[0,0,0,1],cache)
    assert base==[0,1] and set(pool)=={2,3,4}
    assert votes[0]==3  # two distinct parents beat arbitrarily repeated parent0

def test_protected_vote_slots_and_base_survive_high_scoring_long_tail():
    base=list(range(128));fragments=list(range(128,400));votes=fragments[:]
    selected=select_fragments(base,fragments,votes,{v:float(v) for v in fragments})
    assert selected[:128]==base and selected[128:192]==list(range(128,192))
    assert selected[192:]==list(range(399,335,-1))
    assert len(set(selected))==256

def test_short_pool_padding_has_declared_cost_and_same_support():
    selected=select_fragments([7,8],[9],[9],{9:.5})
    assert len(selected)==256 and selected[:3]==[7,8,9] and set(selected)=={7,8,9}
