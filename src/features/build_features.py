"""
PHIL-TEXT: Faz 3.4 — Feature Engineering Modülü
İstatistiksel metin özellikleri, TF-IDF, label encoding,
train/val/test bölme ve feature kaydetme.
"""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

# NLTK kaynakları
for res in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(res, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# ── 1. İstatistiksel Özellikler ──────────────────────────────────────────────

def _sentences(text: str) -> list:
    return [s for s in re.split(r"[.!?]+", text) if len(s.split()) > 2]


def _clean_tokens(text: str) -> list:
    return [t for t in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if t not in STOP_WORDS]


def compute_statistical_features(df: pd.DataFrame,
                                  text_col: str = "text") -> pd.DataFrame:
    """
    Her metin için 20+ istatistiksel özellik türet.

    Özellikler:
    - word_count, char_count, sent_count
    - avg_word_len, avg_sent_len, med_sent_len, std_sent_len
    - unique_word_ratio (TTR)
    - stopword_ratio
    - punct_density  (noktalama / kelime)
    - digit_ratio
    - uppercase_ratio
    - long_word_ratio  (>7 harf)
    - paragraph_count
    - avg_paragraph_len
    - hapax_ratio  (sadece 1 kez geçen kelime oranı)
    - vocab_richness (MTLD proxy: unique / sqrt(total))
    """
    df = df.copy()
    records = []

    for _, row in df.iterrows():
        text = str(row[text_col])
        words_raw = text.split()
        words_lower = [w.lower() for w in words_raw]
        tokens_clean = _clean_tokens(text)
        sents = _sentences(text)
        sent_lens = [len(s.split()) for s in sents]
        paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 20]
        punct_chars = sum(1 for c in text if c in ".,;:!?-\"'()")
        digits = sum(c.isdigit() for c in text)
        upper_words = sum(1 for w in words_raw if w.isupper() and len(w) > 1)
        long_words = sum(1 for w in words_lower if len(w) > 7)
        word_freq = pd.Series(tokens_clean).value_counts() if tokens_clean else pd.Series(dtype=int)
        hapax = (word_freq == 1).sum()

        n = len(words_raw)
        n_clean = len(tokens_clean)
        n_unique = len(set(words_lower))

        rec = {
            "word_count":        n,
            "char_count":        len(text),
            "sent_count":        len(sents),
            "avg_word_len":      round(np.mean([len(w) for w in words_lower]) if words_lower else 0, 3),
            "avg_sent_len":      round(np.mean(sent_lens) if sent_lens else 0, 3),
            "med_sent_len":      round(np.median(sent_lens) if sent_lens else 0, 3),
            "std_sent_len":      round(np.std(sent_lens) if len(sent_lens) > 1 else 0, 3),
            "unique_word_ratio": round(n_unique / n if n else 0, 5),  # TTR
            "stopword_ratio":    round((n - n_clean) / n if n else 0, 5),
            "punct_density":     round(punct_chars / n if n else 0, 5),
            "digit_ratio":       round(digits / len(text) if text else 0, 6),
            "uppercase_ratio":   round(upper_words / n if n else 0, 5),
            "long_word_ratio":   round(long_words / n if n else 0, 5),
            "paragraph_count":   len(paragraphs),
            "avg_para_len":      round(np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0, 2),
            "hapax_ratio":       round(hapax / n_clean if n_clean else 0, 5),
            "vocab_richness":    round(n_unique / np.sqrt(n) if n else 0, 3),
        }
        records.append(rec)

    feat_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df.drop(columns=[c for c in feat_df.columns if c in df.columns], errors="ignore"),
                        feat_df], axis=1)
    logger.info(f"Istatistiksel ozellikler uretildi: {len(feat_df.columns)} ozellik, {len(result)} satir")
    return result


# ── 2. Label Encoding ─────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> dict:
    """
    philosopher, era, school sütunlarını sayısal kodla.
    Returns dict with encoded df + mapping dicts.
    """
    df = df.copy()
    encoders = {}
    mappings = {}

    for col in ["philosopher", "era", "school"]:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[f"{col}_id"] = le.fit_transform(df[col])
        encoders[col] = le
        mappings[col] = {name: int(idx) for idx, name in enumerate(le.classes_)}
        logger.info(f"  {col}: {len(le.classes_)} sinif kodlandi")

    return {"df": df, "encoders": encoders, "mappings": mappings}


# ── 3. TF-IDF Özellikleri ─────────────────────────────────────────────────────

def build_tfidf_features(texts: list, max_features: int = 5_000,
                          ngram_range: tuple = (1, 2),
                          sublinear_tf: bool = True) -> tuple:
    """
    TF-IDF matrisini oluştur.
    Returns: (sparse_matrix, vectorizer)
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        min_df=2,
    )
    X = vec.fit_transform(texts)
    logger.info(f"TF-IDF: {X.shape[0]} dok × {X.shape[1]} ozellik")
    return X, vec


# ── 4. Train / Val / Test Bölme ───────────────────────────────────────────────

def split_dataset(df: pd.DataFrame,
                  text_col: str = "text",
                  label_col: str = "philosopher_id",
                  test_size: float = 0.15,
                  val_size: float = 0.15,
                  random_state: int = 42) -> dict:
    """
    Stratified train/val/test bölme.
    Returns dict with split arrays.
    """
    X = df[text_col].values
    y = df[label_col].values

    # Önce test ayır
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Sonra val ayır
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac,
        stratify=y_trainval, random_state=random_state)

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    logger.info(f"Bolme: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return splits


# ── 5. Özellik Ölçekleme ─────────────────────────────────────────────────────

STAT_FEATURE_COLS = [
    "word_count", "char_count", "sent_count", "avg_word_len", "avg_sent_len",
    "med_sent_len", "std_sent_len", "unique_word_ratio", "stopword_ratio",
    "punct_density", "digit_ratio", "uppercase_ratio", "long_word_ratio",
    "paragraph_count", "avg_para_len", "hapax_ratio", "vocab_richness",
]


def scale_features(df_train: pd.DataFrame,
                   df_val: pd.DataFrame,
                   df_test: pd.DataFrame,
                   feature_cols: list = None) -> tuple:
    """StandardScaler ile istatistiksel özellikleri ölçekle."""
    if feature_cols is None:
        feature_cols = [c for c in STAT_FEATURE_COLS if c in df_train.columns]
    scaler = StandardScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])
    logger.info(f"Olceklendi: {len(feature_cols)} ozellik")
    return df_train, df_val, df_test, scaler


# ── 6. Kaydetme ───────────────────────────────────────────────────────────────

def save_features(splits: dict, label_mappings: dict,
                  output_dir: str = "data/processed") -> None:
    """Split'leri .npz, mapping'leri JSON olarak kaydet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "splits.npz",
        X_train=splits["X_train"], y_train=splits["y_train"],
        X_val=splits["X_val"],   y_val=splits["y_val"],
        X_test=splits["X_test"], y_test=splits["y_test"],
    )
    with open(out / "label_mappings.json", "w", encoding="utf-8") as f:
        json.dump(label_mappings, f, ensure_ascii=False, indent=2)
    logger.info(f"Ozellikler kaydedildi: {out}")


def save_stat_features(df: pd.DataFrame,
                        output_dir: str = "data/processed") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text"], errors="ignore").to_parquet(
        out / "corpus_features.parquet", index=False)
    logger.info(f"Istatistiksel ozellik tablosu kaydedildi: {out/'corpus_features.parquet'}")
