import base64
import io
import pandas as pd
import matplotlib.pyplot as plt

def df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False)

def plot_psi_bar(df: pd.DataFrame) -> str:
    fig = plt.figure()
    plt.bar(df["feature"].astype(str).tolist(), df["psi"].astype(float).tolist())
    plt.xticks(rotation=90)
    plt.ylabel("PSI")
    plt.title("Feature Drift (PSI)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_html(drift_df: pd.DataFrame) -> str:
    img = plot_psi_bar(drift_df.head(20))
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Drift Report</title></head>
<body>
<h1>Drift Report (PSI)</h1>
<p>Top 20 features com maior PSI.</p>
<img src="{img}" />
<h2>Tabela completa</h2>
{df_to_html(drift_df)}
</body>
</html>"""
    return html
