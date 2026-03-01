import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Lista explícita do que é numérico (evita tratar "Fase" como numérico)
NUMERIC_FEATURES = {
    "Idade", "Ano_ingresso", "IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "IAN", "INDE", "tempo_PM"
}

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = [c for c in X.columns if c in NUMERIC_FEATURES]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])
