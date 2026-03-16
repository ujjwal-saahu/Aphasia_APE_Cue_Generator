import re
import unicodedata
import numpy as np
import pandas as pd


RAW_TO_STANDARD_COLUMNS = {
    "詞性": "pos",
    "種類": "main_category",
    "細項": "subcategory",
    "目標詞彙": "target_word",
    "提示": "cue_text",
    "評分區(林治療師評分區)": "score_lin",
    "評分區(王治療師評分區)": "score_wang",
    "評分區(邱治療師評分區)": "score_chiu",
}


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_score(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = unicodedata.normalize("NFKC", value)
    match = re.search(r"-?\d+(\.\d+)?", value)
    if match:
        return float(match.group())
    return np.nan


def load_and_clean_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RAW_TO_STANDARD_COLUMNS)

    text_cols = ["pos", "main_category", "subcategory", "target_word", "cue_text"]
    for col in text_cols:
        df[col] = df[col].apply(normalize_text)

    score_cols = ["score_lin", "score_wang", "score_chiu"]
    for col in score_cols:
        df[col] = df[col].apply(normalize_score)

    df = df.dropna(subset=["target_word", "cue_text"]).copy()

    df["score_mean"] = df[score_cols].mean(axis=1)
    df["score_std"] = df[score_cols].std(axis=1)
    df["cue_length_chars"] = df["cue_text"].str.len()

    df = df.drop_duplicates(
        subset=["pos", "main_category", "subcategory", "target_word", "cue_text"]
    ).reset_index(drop=True)

    return df


def rank_cues_within_word(df):
    df = df.copy()
    df["cue_rank_within_word"] = (
        df.groupby("target_word")["score_mean"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return df


def build_best_human_cues(df):
    ranked = rank_cues_within_word(df)
    best_df = (
        ranked.sort_values(
            ["target_word", "score_mean", "cue_length_chars"],
            ascending=[True, False, True]
        )
        .groupby("target_word", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_df