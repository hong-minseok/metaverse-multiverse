# Associations of autobiographical speech embeddings with depressive symptoms and self-esteem: A multiverse analysis across 50 language models

Analysis code for the revised manuscript. The example contains paired fictional narratives and questionnaire responses for 30 participants.

```text
01_preprocessing.py    Clean paired narratives and score PHQ-9 and RSES
02_embeddings.py       Generate speech embeddings
03_measures.py         Measure speech, embedding distances, condition AUC, and alignment
04_analysis.R          Estimate associations before and after linguistic adjustment, decompose changes, compare model characteristics, and vary the minimum speech length
models.json            Embedding model settings and characteristics
example/speeches.xlsx  Fictional input data
```

The CPU example uses Python 3.12 with pandas, openpyxl, NumPy, SciPy, scikit-learn, kiwipiepy 0.23.2, PyTorch 2.13.0, Transformers 4.56.2, and Sentence Transformers 5.1.1. The analysis uses R 4.6.1 with boot, jsonlite, and pima 1.2.1.

Save `SentiWord_info.json` in this directory, then run the four commands below. [KNU Korean Sentiment Lexicon](https://github.com/park1200656/KnuSentiLex/blob/c95a8a9bcb78ef92e3f8ddd277abaf31451d9f23/data/SentiWord_info.json)

```sh
python 01_preprocessing.py example/speeches.xlsx data/participants.csv
python 02_embeddings.py data/participants.csv data/embeddings minilm ko-sbert e5-base --device cpu
python 03_measures.py data/participants.csv data/embeddings SentiWord_info.json data/measures
Rscript 04_analysis.R data/measures results models.json
```

For study data, pass `--reviewed-ids path/to/included.csv` to step 03 (one `sample_id` per row). `data/measures/` includes alignment, standardized SED, between-model SD, and speech-length contrasts. `results/` contains model estimates, joint tests and intervals before and after linguistic adjustment, Gelbach contributions, participant-bootstrap model comparisons, and age- and sex-adjusted results by minimum speech length (2–20).

[MIT License](LICENSE). Copyright © 2026 Hong Minseok.
