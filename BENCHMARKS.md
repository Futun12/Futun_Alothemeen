# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.153 | 1.510 | 27.0 | 25.0 | 0.0045 |
| XLM-R | 1.672 | 1.434 | 21.0 | 23.0 | 0.0000 |
| CAMeLBERT | 1.405 | 2.705 | 20.0 | 38.0 | 0.0080 |
| DistilBERT | 4.527 | 1.298 | 47.0 | 21.0 | 0.0022 |

- Golden preprocessing: ___ / 25 passed
- PII masking recall: ___ / 60 = ___%
## Lab 2 — Attention-Map Diagnostics

| Diagnostic | Finding |
|---|---|
| Adjacency-looking head | Head 7 on the Arabic Bayan example had the strongest adjacency pattern, with an average adjacent-token attention score of 0.186913. |
| [SEP] sink behaviour | Head 2 showed the strongest [SEP] sink behaviour, with average attention to [SEP] of 0.188075 on the Arabic example. |
| PAD mass with correct mask | 0.000000 |
| PAD mass without mask | 0.043562 |
| Pad leakage prevented | Yes |

The correct attention mask reduced the average attention mass assigned to padded positions from 0.043562 to 0.000000.
## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
