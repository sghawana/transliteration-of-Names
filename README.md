Note: repository in progress...

---

# Transliteration of Names

### Overview

This project focuses on transliterating names from English to Hindi using various sequence-to-sequence (seq2seq) models. The transliteration task involves converting the phonetic sounds of English names into their equivalent Hindi script representations. This is essential for preserving the pronunciation and cultural context of names across languages.

---
### Seq2Seq Models

A standard seq2seq model consists of two main components:

•	**Encoder**: Processes the input sequence (in this case, an English name) and converts it into a context or hidden representation. This representation captures the meaning and structure of the input, which the model then uses to generate the output sequence.

•	**Decoder**: Takes the encoder’s representation and generates the output sequence step-by-step (in this case, the transliterated Hindi name).


