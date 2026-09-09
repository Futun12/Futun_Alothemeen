# Bayan | بيان

Bayan is a bilingual Natural Language Processing (NLP) project designed to analyze and understand citizen feedback written in both Arabic and English. The project applies different NLP techniques and transformer-based models to process text and extract useful information.

The project includes a bilingual preprocessing pipeline for Arabic and English text. The preprocessing handles text normalization, unnecessary characters, repeated characters, sentence segmentation, and masking sensitive information such as phone numbers and national IDs. Different tokenizers, including mBERT, XLM-RoBERTa, CAMeLBERT, and DistilBERT, were also compared to evaluate their performance on Arabic and English text.

The project also explores transformer architecture by implementing scaled dot-product attention and multi-head attention. Attention maps are analyzed to understand how transformer models attend to different tokens and how attention masks prevent attention to padding tokens.

For topic classification, a baseline model using TF-IDF and LinearSVC is implemented and compared with a transformer-based classifier using XLM-RoBERTa. The dataset is split using citizen groups to prevent data leakage between training, validation, and testing data.

Bayan also includes Named Entity Recognition (NER) to identify useful information in citizen feedback, such as services, locations, dates, reference numbers, and organizations. BIO labels are aligned with transformer subword tokens to support token-level classification.

The project also includes extractive Question Answering (QA), where the model identifies an answer span from a given context. The QA component supports no-answer cases, allowing the model to return no answer when the requested information is not available in the context.

The project uses Python, PyTorch, Hugging Face Transformers, scikit-learn, spaCy, XLM-RoBERTa, mBERT, CAMeLBERT, and other NLP tools for text processing, model development, testing, and evaluation.
https://github.com/SDAIAAcademy
