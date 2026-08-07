# mmlu-pt

Pipeline para criar um dataset MCQA em português com validação fail-closed e
deduplicação exact/fuzzy do NVIDIA NeMo Curator.

## Ambiente

Requisitos: Python 3.12, Linux x86_64, GPU NVIDIA e CUDA 12 compatível.

```bash
uv sync
```

O projeto usa `nemo-curator[text_cuda12]==1.3.0`. O vLLM é fixado em 0.15.1
porque releases posteriores dependem de um wheel de `xgrammar` indisponível para
Python 3.12/Linux x86_64.

## Sources

[`config/sources.yaml`](config/sources.yaml) contém um caminho relativo exato por
CSV, além de `exam` e `academic_level`. Não há descoberta recursiva nem globs.
AFA, COMVEST e FUVEST estão comentados porque não atendem ao schema obrigatório;
o motivo está registrado ao lado de cada entrada.

Colunas obrigatórias nos CSVs ativos:

```text
exam_edition, exam_url, num_questions, num,
statement, alternatives, answer, url
```

`subject` e demais colunas extras são ignorados. `question_num` e
`num_question` não são tratados como aliases.

## Execução

Use sempre um diretório de saída novo ou vazio:

```bash
uv run mmlu-pt \
  --sources config/sources.yaml \
  --output-dir runs/mcqa
```

Saídas principais:

```text
runs/mcqa/
├── dataset/                         # Parquet público final
├── audit/schema_validation.json
├── audit/answer_conflicts.parquet
├── audit/run_summary.json
├── dedup/                           # artefatos exact/fuzzy do NeMo
└── staging/                         # Parquets intermediários auditáveis
```

O comando termina antes de criar os estágios quando algum source falha no
preflight. Todos os erros de schema são agregados no mesmo relatório.

## Testes

Testes CPU, executados por padrão:

```bash
uv run pytest
```

Preflight do snapshot externo:

```bash
uv run pytest -m acceptance tests/test_acceptance.py
```

Esse grupo reproduz, em ordem, a primeira passagem com os 25 paths (esperando o
relatório agregado de AFA, sete COMVEST e FUVEST) e a segunda com os 16 sources
ativos do manifesto versionado. O teste cria o manifesto temporário de 25
entradas sem modificar `config/sources.yaml`.

Pipeline sintético com GPU:

```bash
uv run pytest -m gpu tests/test_gpu_pipeline.py
```

Aceitação completa com os CSVs reais e GPU:

```bash
uv run pytest -m 'acceptance and gpu' tests/test_acceptance.py::test_real_pipeline_snapshot
```
