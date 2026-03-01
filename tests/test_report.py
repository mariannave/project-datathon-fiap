import pandas as pd

from src.report import build_html, df_to_html, plot_psi_bar


def test_df_to_html_contains_table_markup():
    df = pd.DataFrame({"feature": ["A"], "psi": [0.1]})
    html = df_to_html(df)
    assert "<table" in html
    assert "feature" in html


def test_plot_psi_bar_returns_base64_data_uri():
    df = pd.DataFrame({"feature": ["A", "B"], "psi": [0.1, 0.2]})
    img = plot_psi_bar(df)
    assert img.startswith("data:image/png;base64,")


def test_build_html_contains_expected_sections():
    df = pd.DataFrame({"feature": ["A"], "type": ["numeric"], "psi": [0.1]})
    html = build_html(df)
    assert "Drift Report (PSI)" in html
    assert "Tabela completa" in html
    assert "<img src=\"data:image/png;base64," in html
