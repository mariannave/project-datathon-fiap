import pandas as pd
from src.model import build_pipeline

def test_pipeline_predict_proba_shape():
    X = pd.DataFrame({
        "Idade":[10,11,12],
        "Genero":["M","F","M"],
        "IAA":[7.0,6.0,8.0],
        "IEG":[5.0,6.0,7.0],
    })
    y = [0,1,0]
    pipe = build_pipeline(X)
    pipe.fit(X,y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (3,2)
