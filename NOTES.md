# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Sentence Segmentation Spot Check

- English example: `The road is damaged. Please fix it soon.`
- Output: `['The road is damaged.', 'Please fix it soon.']`
- Finding: The pipeline correctly split the English text into two sentences.

- Arabic example: `الطريق متضرر. نرجو إصلاحه قريباً.`
- Output: `['الطريق متضرر.', 'نرجو إصلاحه قريباً.']`
- Finding: The pipeline correctly split the Arabic text into two sentences.

- PII example: `رقمي 0551234567. الرجاء التواصل معي.`
- Output: `['رقمي <PHONE>.', 'الرجاء التواصل معي.']`
- Finding: Preprocessing masked the phone number before sentence segmentation.



### Defect 1
- Class: PII
- Example: `0551234567` and `1023456789`
- Why it matters: Personal information should not be exposed to the model or stored unnecessarily.
- Decision: Clean / mask as `<PHONE>` and `<NATIONAL_ID>`.

### Defect 2
- Class: HTML remnants
- Example: `<br>`
- Why it matters: HTML tags are not part of the actual citizen feedback and add noise.
- Decision: Clean / remove the HTML tags.

### Defect 3
- Class: Leading and trailing whitespace
- Example: `  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   `
- Why it matters: Extra spaces make the text inconsistent.
- Decision: Clean / remove extra leading and trailing spaces.

### Defect 4
- Class: Emoji
- Example: `😡`
- Why it matters: Emojis may contain useful sentiment information.
- Decision: Preserve because it may help sentiment classification.

### Defect 5
- Class: Repeated letters / elongation
- Example: `لووووسمحت`
- Why it matters: Repeated letters can create unnecessary token fragmentation, but may also express emphasis.
- Decision: Task-dependent.

### Defect 6
- Class: Arabic orthographic variation
- Example: `ألطريق`, `ألإنارة`, `أحتساب`
- Why it matters: Different spellings of similar words can reduce consistency for tokenisation and model input.
- Decision: Task-dependent / normalise carefully.

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
