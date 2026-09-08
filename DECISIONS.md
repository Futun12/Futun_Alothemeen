# Decision Records

## tokenizer
- Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)
- Arabic fertility evidence: XLM-R achieved an Arabic fertility of 1.672. CAMeLBERT was better for Arabic alone at 1.405, while mBERT scored 2.153 and DistilBERT scored 4.527.
- English fertility evidence: XLM-R achieved an English fertility of 1.434, close to the best result from DistilBERT at 1.298, and better than mBERT at 1.510 and CAMeLBERT at 2.705.
- p95 length evidence: XLM-R had an Arabic p95 sequence length of 21 and an English p95 sequence length of 23, giving short and balanced sequence lengths across both languages.
- Operational trade-off / rationale: XLM-R was chosen because Bayan is a bilingual Arabic-English system. CAMeLBERT performed best for Arabic but was much less efficient for English, while DistilBERT performed best for English but fragmented Arabic heavily. XLM-R provides the strongest balance across both languages and also achieved a 0.0000 Arabic UNK rate in this audit.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
