# Projeto de MLE: Modelo de Risco de Defasagem Escolar

## Escopo deste documento

Este arquivo é a documentação técnico-gerencial para avaliação acadêmica (decisões de modelagem, métricas, trade-offs e limitações).

Guia operacional rápido (instalação e comandos do dia a dia):
- `README.md`

## 1. Resumo executivo

Este projeto implementa uma solução ponta a ponta de Machine Learning para estimar risco de defasagem escolar no contexto do Datathon PEDE (Associação Passos Mágicos).

Entrega técnica:

1. Pipeline de treino e validação temporal.
2. Serialização do modelo (`artifacts/model.joblib`).
3. API de inferência (`FastAPI`, endpoint `POST /predict`).
4. Logging de inferências para monitoramento.
5. Relatório de métricas e monitoramento de drift.
6. Política de decisão conservadora para reduzir falso positivo.

Objetivo operacional:

- transformar os indicadores do aluno em uma probabilidade de risco (`risk_probability`) e uma decisão (`risk_label`) adequada à estratégia de negócio.

## 2. Problema de negócio

O problema tratado é binário:

- classe `1` (risco): estudante com `Defasagem > 0`;
- classe `0` (sem risco): estudante com `Defasagem <= 0`.

No cenário educacional, o custo de erro depende da estratégia:

1. Estratégia de triagem: prioriza identificar todos os casos de risco (recall alto).
2. Estratégia conservadora: prioriza reduzir alarmes falsos (precision maior e menos falso positivo).

Este projeto adotou política conservadora na API para reduzir falso positivo em produção.

## 3. Arquitetura da solução

Componentes principais:

1. `src/train.py`: treinamento e geração de artefatos.
2. `src/feature_engineering.py`: limpeza, engenharia e seleção de atributos.
3. `src/model.py`: pipeline de preprocessamento + classificador.
4. `api/main.py`: API de predição com política de thresholds configuráveis.
5. `src/drift_cli.py` + `src/drift.py`: cálculo de drift (PSI) e relatório.

Artefatos gerados:

1. `artifacts/model.joblib`
2. `artifacts/reference_stats.json`
3. `reports/metrics.json`
4. `reports/confusion_matrix.png`
5. `reports/drift_report.html`
6. `reports/threshold_analysis.csv`

## 4. Dados utilizados e protocolo de treino

Base:

1. PEDE 2022
2. PEDE 2023
3. PEDE 2024

Estratégia temporal:

1. Treino: features de 2022 para prever alvo em 2023.
2. Validação: features de 2023 para prever alvo em 2024.

Esse desenho reduz risco de leakage temporal em comparação a split aleatório puro.

## 5. Etapas do Pipeline de Machine Learning

Para aderência explícita ao requisito do Datathon, o pipeline está organizado nas seguintes etapas:

1. Pré-processamento dos Dados: normalização de nomes de colunas, tratamento de nulos, coerção de tipos numéricos e padronização de categorias.
2. Engenharia de Features: construção de atributos derivados (ex.: `tempo_PM`) e seleção de variáveis estáveis entre anos.
3. Treinamento e Validação: treino temporal com 2022->2023 e validação temporal com 2023->2024, reduzindo risco de leakage.
4. Seleção de Modelo: uso de pipeline com imputação, codificação categórica e `LogisticRegression`, escolhido por equilíbrio entre desempenho e interpretabilidade.
5. Pós-processamento (aplicável): conversão de probabilidade em decisão via thresholds de negócio (`risk_threshold` e `review_threshold`) e classificação por faixa de risco.

## 6. Engenharia de atributos

Features estáveis utilizadas:

1. Categóricas: `Fase`, `Turma`, `Genero`, `Instituicao_de_ensino`, `Pedra_atual`, `Pedra_anterior`, `Atingiu_PV`, `Indicado`.
2. Numéricas: `Idade`, `Ano_ingresso`, `IAA`, `IEG`, `IPS`, `IPP`, `IDA`, `IPV`, `IAN`, `INDE`, `tempo_PM`.

Tratamentos relevantes:

1. Renomeação/normalização de colunas com variações entre planilhas.
2. Conversão robusta de campos numéricos que chegam com artefato de Excel (ex.: `1900-01-07 00:00:00`).
3. Criação de `tempo_PM` (tempo no programa).
4. Preenchimento de categóricos ausentes com `UNKNOWN`.

## 7. Modelo e pipeline

Pipeline (`src/model.py`):

1. Numéricas: `SimpleImputer(strategy="median")`.
2. Categóricas: `SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")`.
3. Classificador: `LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")`.

Racional:

1. Boa interpretabilidade e baseline forte para dados tabulares.
2. `class_weight="balanced"` ajuda com desbalanceamento entre classes.

## 8. Métricas do modelo (validação 2023->2024)

Fonte: `reports/metrics.json`.

1. `roc_auc`: **0.8635**
2. `accuracy`: **0.6170**
3. Classe de risco (1): `precision=0.2595`, `recall=0.9808`, `f1=0.4105`, `support=104`
4. Classe sem risco (0): `precision=0.9946`, `recall=0.5598`, `f1=0.7164`, `support=661`

Leitura executiva:

1. O score (`AUC`) separa bem as classes.
2. Com threshold padrão de classificação baixo, o modelo encontra quase todos os casos de risco.
3. O custo é alto número de falso positivo.

## 9. Política de decisão conservadora (API)

A API retorna probabilidade e aplica política de decisão por thresholds:

1. `RISK_THRESHOLD` (default: `0.8`)
2. `REVIEW_THRESHOLD` (default: `0.5`)
3. Regra: `0 <= REVIEW_THRESHOLD <= RISK_THRESHOLD <= 1`

Bandas:

1. `high_risk`: `risk_probability >= risk_threshold`
2. `review`: `review_threshold <= risk_probability < risk_threshold`
3. `low_risk`: `risk_probability < review_threshold`

Decisão final:

- `risk_label = 1` apenas em `high_risk`.

Isso reduz falso positivo e torna a saída mais conservadora.

## 10. Trade-off real de thresholds (modelo atual)

Fonte: `reports/threshold_analysis.csv`.

| threshold | accuracy | precision_risco | recall_risco | false_positives | true_positives |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0.6170 | 0.2595 | 0.9808 | 291 | 102 |
| 0.7 | 0.7908 | 0.3761 | 0.8173 | 141 | 85 |
| 0.8 | 0.8248 | 0.4074 | 0.6346 | 96 | 66 |
| 0.9 | 0.8431 | 0.3750 | 0.2308 | 40 | 24 |

Interpretação:

1. Aumentar threshold reduz falso positivo.
2. Aumentar threshold também reduz recall da classe de risco.
3. O valor `0.8` foi adotado como compromisso conservador.

## 11. Contrato da API (`POST /predict`)

Payload (campos principais):

1. `year` (obrigatório para rastreabilidade; removido antes da predição).
2. Demais atributos de contexto e indicadores (`IAA`, `IEG`, `INDE`, etc.).

Resposta:

```json
{
  "risk_label": 0,
  "risk_probability": 0.0341,
  "risk_band": "low_risk",
  "risk_threshold": 0.8,
  "review_threshold": 0.5
}
```

## 12. O que esperar na prática

1. `risk_probability` muda conforme o modelo em `artifacts/model.joblib`.
2. `risk_label` depende da política de threshold ativa.
3. O mesmo payload pode gerar resposta diferente após novo treino.

## 13. Reprodutibilidade e execução

Fluxo recomendado:

```bash
source .venv/bin/activate
python -m src.train --excel data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx --out artifacts
RISK_THRESHOLD=0.8 REVIEW_THRESHOLD=0.5 python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 14. Monitoramento

Monitoramento implementado:

1. Log de inferência em `data/processed/inference_log.csv`.
2. Estatísticas de referência de treino em `artifacts/reference_stats.json`.
3. Drift por PSI em `reports/drift_report.html`.

Comando:

```bash
python -m src.drift_cli --reference artifacts/reference_stats.json --inference data/processed/inference_log.csv --out reports/drift_report.html
```

## 15. Qualidade de software e cobertura de testes

O projeto adota critério mínimo de **80% de cobertura** sobre módulos de aplicação (`src` e `api`), configurado em `.coveragerc` (`fail_under = 80`).

Comandos para evidência:

```bash
source .venv/bin/activate
python -m coverage erase
python -m coverage run -m pytest -q
python -m coverage report -m
python -m coverage html
```

Interpretação esperada:

1. O comando `coverage report -m` deve mostrar `TOTAL >= 80%`.
2. Em caso de cobertura abaixo do mínimo, a execução retorna erro (não conformidade).

## 16. Limitações e próximos passos

Limitações atuais:

1. Base limitada e desbalanceada.
2. Ausência de labels online para monitorar performance em produção em tempo real.

Próximos passos recomendados:

1. Calibração de probabilidade.
2. Otimização automática de threshold por objetivo de negócio (ex.: precisão mínima).
3. Validação temporal com mais janelas quando houver novos anos.
