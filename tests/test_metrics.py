import numpy as np
from phora.metrics import fast_challenge_score

def test_perfect_score():
    y=np.array(["LR-1","LR-2","LR-3","LR-4","LR-5","LR-M","LR-TIV"],dtype=object)
    m=fast_challenge_score(y,y)
    assert abs(m["final_score"]-1.0) < 1e-8
    assert abs(m["special_category_recognition"]-1.0) < 1e-8
