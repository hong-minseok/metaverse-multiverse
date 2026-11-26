import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score

MATTR_WINDOW = 10
CONTENT_TAGS = set("NNB NNG NNP NP NR VA VV".split())
UNCOUNTED_TAGS = set("SB SE SF SO SP SS SSC SSO SW UN Z_CODA Z_SIOT".split())
CONJUGATION_SUFFIX = re.compile(r"(?:-[IR])+$")
NEGATIONS = {("안", "MAG"), ("못", "MAG"), ("않", "VX"), ("아니", "VCN")}
METRICS = ("cosine", "euclidean", "angular", "l1")


def surface(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def sentiment_lexicon(path):
    words = {surface(row["word"]): int(row["polarity"]) for row in json.loads(path.read_text())
             if int(row["polarity"]) and surface(row["word"])}
    by_initial = defaultdict(list)
    for word in sorted(words, key=len, reverse=True):
        by_initial[word[0]].append((word, np.sign(words[word])))
    return by_initial


def sentiment_counts(text, lexicon):
    counts = Counter()
    text = surface(text)
    start = 0
    while start < len(text):
        for word, sign in lexicon[text[start]]:
            if text.startswith(word, start):
                counts[sign] += 1
                start += len(word)
                break
        else:
            start += 1
    return counts[1], counts[-1]


def speech_features(text, kiwi, lexicon):
    morphs = [(unicodedata.normalize("NFC", t.form), CONJUGATION_SUFFIX.sub("", t.tag))
              for t in kiwi.tokenize(text)]
    content = [m for m in morphs if m[1] in CONTENT_TAGS]
    morphemes = sum(tag not in UNCOUNTED_TAGS and not tag.startswith("W_") for _, tag in morphs)
    negation = 0
    previous = None
    for form, tag in morphs:
        negation += ((form, tag) in NEGATIONS or
                     (tag == "VX" and form in {"말", "못하"} and previous == ("지", "EC")))
        if not tag.startswith("J"):
            previous = (form, tag)
    positive, negative = sentiment_counts(text, lexicon)
    window = min(len(content), MATTR_WINDOW)
    diversity = [len(set(content[i:i + window])) / window
                 for i in range(len(content) - window + 1)] if window else []
    features = dict(
        lexical_diversity=sum(diversity) / len(diversity) if diversity else np.nan,
        net_sentiment=(positive - negative) / morphemes if morphemes else np.nan,
        sentiment_density=(positive + negative) / morphemes if morphemes else np.nan,
        speech_length=np.log(len(content)) if content else np.nan,
        negation_density=100 * negation / morphemes if morphemes else np.nan,
    )
    return content, features


def reviewed_participants(participants, reviewed_ids):
    data = pd.read_csv(participants, dtype={"sample_id": str}).set_index("sample_id")
    if reviewed_ids is not None:
        included = pd.read_csv(reviewed_ids, dtype={"sample_id": str})["sample_id"]
        data = data.loc[included]
    if not data.index.is_unique:
        raise ValueError("Expected one row per participant")
    return data.sort_index()


def paired_features(data, lexicon):
    kiwi = Kiwi(model_type="cong")
    rows = []
    for person in data.itertuples():
        negative_words, negative = speech_features(person.negative, kiwi, lexicon)
        positive_words, positive = speech_features(person.positive, kiwi, lexicon)
        length = min(len(negative_words), len(positive_words))
        if length < 2:
            continue
        negative_counts = Counter(negative_words)
        positive_counts = Counter(positive_words)
        words = sorted(negative_counts.keys() | positive_counts.keys())
        row = dict(sample_id=person.Index, shorter_speech_length=length, lexical_distance=cosine(
            [negative_counts[w] for w in words], [positive_counts[w] for w in words]
        ))
        for feature in negative:
            row[feature + "_mean"] = (negative[feature] + positive[feature]) / 2
            row[feature + "_difference"] = negative[feature] - positive[feature]
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id")


def distances(negative, positive):
    similarity = np.clip(np.einsum("ij,ij->i", negative, positive), -1, 1)
    difference = negative - positive
    return (1 - similarity, np.linalg.norm(difference, axis=1),
            np.arccos(similarity), np.abs(difference).sum(axis=1))


def condition_auc(negative, positive):
    n = len(negative)
    centers = [(v.sum(axis=0) - v) / (n - 1) for v in (negative, positive)]
    scores = [((v - centers[0]) ** 2).sum(axis=1) - ((v - centers[1]) ** 2).sum(axis=1)
              for v in (negative, positive)]
    return roc_auc_score(np.repeat([0, 1], n), np.concatenate(scores))


def condition_alignment(negative, positive):
    difference = positive - negative
    other = difference.sum(axis=0) - difference
    norm = np.linalg.norm(difference, axis=1) * np.linalg.norm(other, axis=1)
    return np.divide(np.einsum("ij,ij->i", difference, other), norm,
                     out=np.full(len(norm), np.nan), where=norm > 0)


def speech_length_measures(length, cosine_sed, alignment, output):
    standardized = (cosine_sed - cosine_sed.mean()) / cosine_sed.std()
    between_model_sd = standardized.std(axis=1)
    paired = pd.concat({"alignment": alignment.stack(), "standardized_sed": standardized.stack()},
                       axis=1).rename_axis(["sample_id", "model_key"]).join(length)
    paired.join(between_model_sd.rename("between_model_sd")).to_csv(output / "speech_length_measures.csv")
    bands = pd.cut(length, [1, 4, 9, 14, 19, np.inf], labels=["2-4", "5-9", "10-14", "15-19", "20+"])
    pd.DataFrame({
        "n": length.groupby(bands, observed=False).size(),
        "alignment": alignment.groupby(bands, observed=False).mean().median(axis=1),
        "between_model_sd": between_model_sd.groupby(bands, observed=False).mean(),
    }).rename_axis("speech_length").to_csv(output / "speech_length_bands.csv")

    short = length.to_numpy() < 10
    alignment = alignment.to_numpy()
    spread = between_model_sd.to_numpy()

    def contrast(indices):
        below, above = indices[short[indices]], indices[~short[indices]]
        if not len(below) or not len(above):
            return np.nan, np.nan
        return (np.median(alignment[below].mean(axis=0) - alignment[above].mean(axis=0)),
                spread[below].mean() - spread[above].mean())

    rng = np.random.default_rng(20260907)
    estimate = contrast(np.arange(len(length)))
    draws = np.array([contrast(rng.integers(len(length), size=len(length))) for _ in range(10000)])
    rows = []
    for measure, value, samples in zip(("alignment", "between_model_sd"), estimate, draws.T):
        samples = samples[np.isfinite(samples)]
        lower, upper = np.quantile(samples, [.025, .975]) if len(samples) else (np.nan, np.nan)
        tails = min(np.count_nonzero(samples <= 0), np.count_nonzero(samples >= 0))
        p = min(1, 2 * (1 + tails) / (1 + len(samples))) if len(samples) else np.nan
        rows.append(dict(measure=measure, below10_minus_from10=value, ci_lower=lower, ci_upper=upper,
                         p=p, valid_resamples=len(samples)))
    pd.DataFrame(rows).to_csv(output / "speech_length_contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("participants", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reviewed-ids", type=Path, help="CSV of included sample_id values after review")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reviewed = reviewed_participants(args.participants, args.reviewed_ids)
    features = paired_features(reviewed, sentiment_lexicon(args.dictionary))
    scores = reviewed.loc[features.index, ["PHQ9", "RSES", "age", "male"]]
    scores.join(features).to_csv(output / "participants.csv")

    panel = json.loads(Path(__file__).with_name("models.json").read_text())
    matrices = {metric: pd.DataFrame(index=features.index) for metric in METRICS}
    length = features["shorter_speech_length"]
    primary = length >= 10
    alignment = pd.DataFrame(index=features.index)
    auc = {}
    for key in panel:
        path = args.embeddings / (key + ".npz")
        if not path.exists():
            continue
        with np.load(path) as vectors:
            positions = pd.Index(vectors["sample_id"]).get_indexer(features.index)
            if (positions < 0).any():
                raise ValueError(f"Missing paired embeddings for {key}")
            negative, positive = (vectors[c][positions].astype(float) for c in ("negative", "positive"))
        for v in (negative, positive):
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        for metric, values in zip(METRICS, distances(negative, positive)):
            matrices[metric][key] = values
        alignment[key] = condition_alignment(negative, positive)
        auc[key] = condition_auc(negative[primary], positive[primary])
    if not auc:
        raise ValueError("No panel embeddings in the input directory")
    for metric, matrix in matrices.items():
        matrix.to_csv(output / (metric + ".csv"))
    speech_length_measures(length, matrices["cosine"], alignment, output)
    matrices["cosine"].loc[primary].corr(method="spearman").to_csv(output / "model_agreement.csv")
    pd.Series(auc, name="auc").rename_axis("model_key").to_csv(output / "condition_auc.csv")
