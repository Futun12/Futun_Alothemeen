# EVALUATION REPORT — Bayan

## Executive headline

The Bayan evaluation shows strong aggregate performance, but the validation errors reveal a concentrated `parks` versus `roads` classification failure.

The most important quality risk is the Arabic MSA `parks` slice, so class-specific and sliced evaluation should be considered alongside aggregate metrics.

## Aggregate metrics

| Metric | Result |
|---|---:|
| Macro-F1 | 0.8333 |
| Accuracy | 0.8750 |
| Accuracy 95% bootstrap CI | [0.8617, 0.8888] |

## Sliced metrics with bootstrap CIs

| Slice type | Slice | n | Macro-F1 | Small slice |
|---|---|---:|---:|---|
| language | ar | 1200 | 0.6000 | No |
| language | en | 1200 | 1.0000 | No |
| dialect | MSA | 1200 | 0.6000 | No |
| dialect | nan | 1200 | 1.0000 | No |
| class | parks | 300 | 0.0000 | No |
| class | roads | 300 | 1.0000 | No |
| class | lighting | 300 | 1.0000 | No |
| class | waste | 300 | 1.0000 | No |
| class | water | 300 | 1.0000 | No |
| class | billing | 300 | 1.0000 | No |
| class | digital_services | 300 | 1.0000 | No |
| class | licensing | 300 | 1.0000 | No |
| length | short | 2400 | 0.8333 | No |

Small slices are flagged because their estimates should not be treated as equally precise as larger slices.

## Behavioural suite

| Test type | Status | Target |
|---|---|---|
| Invariance | Implemented | >= 95% |
| Directional | Implemented | Course behavioural checks |
| Minimum functionality | Implemented | >= 90% |

The behavioural test runner covers invariance, directional behaviour, and minimum-functionality tests.

## Error taxonomy

120 sampled validation errors were reviewed.

The dominant observed error pattern was a confusion between
the `parks` and `roads` classes. The reviewed errors had
`parks` as the true label and `roads` as the predicted label.

The errors were concentrated in Arabic MSA examples and
occurred in both short and medium-length examples.

### Prioritised fixes

1. Improve discrimination between `parks` and `roads`
   training examples.
   - Predicted metric delta: High positive impact.

2. Review ambiguous `parks` examples and annotations.
   - Predicted metric delta: Moderate positive impact.

3. Add targeted Arabic MSA `parks` examples and
   behavioural checks.
   - Predicted metric delta: Moderate positive impact.

## Retrieval quality

Lab 5 retrieval evaluation produced:

| Configuration | Recall@10 | MRR@10 |
|---|---:|---:|
| Bi-encoder | 0.0051 | 0.0064 |
| Cross-encoder rerank | 0.0026 | 0.0015 |

The retrieval experiment also demonstrated that ranking changes when the L2-normalisation contract is violated. Retrieval quality must therefore be assessed with the labelled query set rather than by visually inspecting individual results.

## Known limitations

- Validation errors are concentrated in the `parks` versus `roads` classification boundary.
- The observed classification error pattern is concentrated in Arabic MSA examples.
- Some evaluation slices may contain fewer observations than others, so small-slice estimates are less precise.
- Retrieval Recall@10 and MRR@10 remain low and require further improvement.
- Cross-encoder reranking did not improve retrieval metrics in the current experiment.
- Behavioural tests cover selected expected behaviours and cannot represent every possible production input.
- Perfect scores on some supplied evaluation sets do not guarantee equivalent performance on unseen real-world citizen feedback.

## Model cards

Three model cards are generated in:

`docs/model_cards/`

1. `topic_classifier.md`
2. `dialect_aware.md`
3. `ner.md`
