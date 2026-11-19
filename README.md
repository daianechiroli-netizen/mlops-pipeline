# mlops-pipeline

Autor: Daiane Maria De Genaro Chiroli

Projeto gerado como parte do exercício de modularização do curso de MLOps.

## Estrutura

```
mlops-pipeline/
├── data/
│   └── raw/
│       └── points.csv       # gerado pelo generate_data.py
├── src/
│   ├── data.py              # carregamento e preparação dos dados
│   ├── train.py             # treinamento do modelo
│   ├── evaluate.py          # avaliação do modelo
│   └── predict.py           # predição
├── outputs/
│   └── metrics.json         # gerado pelo main.py
├── generate_data.py
├── main.py
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python generate_data.py
python main.py
```
