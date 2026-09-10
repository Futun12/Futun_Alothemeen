"""Lab 3B: NER label alignment."""


def align_labels(word_ids, word_labels):
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:

        # Special tokens such as [CLS], [SEP], and padding
        if word_id is None:
            aligned_labels.append(-100)

        # First subword of a word
        elif word_id != previous_word_id:
            aligned_labels.append(word_labels[word_id])

        # Additional subwords of the same word
        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels