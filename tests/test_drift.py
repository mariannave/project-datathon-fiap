import numpy as np
import pandas as pd
from src.drift import psi, psi_categorical

def test_psi_zero_when_equal():
    e = np.array([0.2,0.8])
    a = np.array([0.2,0.8])
    assert abs(psi(e,a)) < 1e-9

def test_psi_categorical_positive_when_shift():
    ref = {"A":0.5,"B":0.5}
    cur = pd.Series(["A"]*9 + ["B"])
    v = psi_categorical(ref, cur)
    assert v > 0
