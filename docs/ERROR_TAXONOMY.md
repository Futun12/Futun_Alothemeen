# Error Taxonomy Starter

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect

## Hand-read Error Analysis

120 sampled validation errors were reviewed manually.

### Notes

- A repeated error pattern was observed between the `parks` and `roads` classes.
- The reviewed errors had `parks` as the true label but were predicted as `roads`.
- The observed errors were concentrated in Arabic MSA examples.
- The same confusion appeared across both short and medium-length examples.
- The errors were not distributed across many different predicted classes; the dominant pattern was `parks → roads`.
- This suggests difficulty distinguishing feedback about parks from feedback about roads.
- The training examples and labels for these two classes should be reviewed for possible overlap or ambiguity.

### Error-category histogram

The dominant applicable category is **Label ambiguity**, because the observed errors indicate confusion between the `parks` and `roads` topic labels.

| Error category | Observed frequency |
|---|---|
| Label ambiguity | Dominant |
| Arabic orthographic variation | Not observed as a dominant cause |
| Dialect or code-switching | Not observed as a dominant cause |
| Entity boundary or clitic alignment | Not observed |
| Long-context truncation | Not observed |
| Retrieval relevance mismatch | Not applicable to these classification errors |
| Preprocessing or serving skew | Not observed |
| Annotation defect | No clear evidence observed |

### Top 3 Prioritised Fixes

1. Review and improve the distinction between `parks` and `roads` training examples.
   - Predicted metric delta: High positive impact.

2. Review ambiguous `parks` examples and their annotations to ensure that the correct topic label is consistently applied.
   - Predicted metric delta: Moderate positive impact.

3. Add more targeted Arabic MSA examples that clearly distinguish `parks` from `roads`.
   - Predicted metric delta: Moderate positive impact.