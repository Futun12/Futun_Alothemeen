# Model Card — Bayan NER Model

## Intended use
Extract named entities from Bayan Arabic and English feedback.

## Artefact / data versions
- Model/checkpoint: Bayan NER checkpoint
- Preprocessing version: 1.2.0
- Data version/snapshot: Bayan segmented NER evaluation set

## Metrics
| Metric | Result |
|---|---:|
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |

## Slice metrics
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

## Behavioural tests
| Test type | Status | Target |
|---|---|---|
| Invariance | Implemented | >= 95% |
| Directional | Implemented | Course behavioural checks |
| Minimum functionality | Implemented | >= 90% |

## Known limitations
<!-- Lab 6: write this section by hand. Do not auto-generate it. -->
TODO

## Contact / owner
TODO