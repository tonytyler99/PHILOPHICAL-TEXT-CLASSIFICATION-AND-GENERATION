"""
PHIL-TEXT: Faz 3.3 — Veri Temizleme Modülü
Project Gutenberg metinlerinden artefakt temizliği, duplikasyon,
outlier tespiti ve metin kalite kontrolleri.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


# ── Gutenberg başlık / son artefaktları ───────────────────────────────────────
_HEADER_PATTERNS = [
    r"(?i)^\s*the\s+project\s+gutenberg\s+ebook.*?(\*{3}.*?START.*?\*{3})",
    r"(?i)\*{3}\s*START\s+OF\s+(THE|THIS)\s+PROJECT\s+GUTENBERG.*?\*{3}",
    r"(?i)produced\s+by\s+.{0,200}?\n",
    r"(?i)This\s+eBook\s+is\s+for\s+the\s+use\s+of\s+anyone.{0,500}",
]
_FOOTER_PATTERNS = [
    r"(?i)\*{3}\s*END\s+OF\s+(THE|THIS)\s+PROJECT\s+GUTENBERG.*",
    r"(?i)End\s+of\s+(the\s+)?Project\s+Gutenberg.*",
]

# Temizlenecek Unicode / boş karakterler
_UNICODE_NOISE = [
    ("\ufeff", ""),   # BOM
    ("\u00ad", ""),   # soft hyphen
    ("\u2014", " "),  # em-dash
    ("\u2013", " "),  # en-dash
    ("\u2018", "'"), ("\u2019", "'"),  # curly quotes
    ("\u201c", '"'), ("\u201d", '"'),
]


def remove_gutenberg_artifacts(text: str) -> str:
    """Gutenberg başlık, telif ve sondaki metinleri sil."""
    # Footer önce (daha güvenilir)
    for pat in _FOOTER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    # Header
    for pat in _HEADER_PATTERNS:
        match = re.search(pat, text, flags=re.DOTALL)
        if match:
            text = text[match.end():]
            break
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Fazla boşluk, satır sonu, sekme karakterlerini temizle."""
    text = re.sub(r"\r\n|\r", "\n", text)   # Windows satır sonu
    text = re.sub(r"\n{3,}", "\n\n", text)  # çoklu boş satır → max 2
    text = re.sub(r"[ \t]{2,}", " ", text)  # çoklu boşluk → tek
    return text.strip()


def replace_unicode_noise(text: str) -> str:
    for old, new in _UNICODE_NOISE:
        text = text.replace(old, new)
    return text


def clean_text_pipeline(text: str) -> str:
    """Tam temizleme boru hattı."""
    text = replace_unicode_noise(text)
    text = remove_gutenberg_artifacts(text)
    text = normalize_whitespace(text)
    return text


def detect_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Tam veya yakın duplikat satırları işaretle."""
    df = df.copy()
    # Tam duplikat (aynı filozof + aynı eser adı)
    df["is_exact_dup"] = df.duplicated(subset=["philosopher", "work"], keep="first")
    # İlk 500 karakter benzerliği (yakın duplikat proxy)
    df["text_head"] = df[text_col].str[:500]
    df["is_near_dup"] = df.duplicated(subset=["text_head"], keep="first")
    df.drop(columns=["text_head"], inplace=True)
    n_exact = df["is_exact_dup"].sum()
    n_near = df["is_near_dup"].sum()
    if n_exact: logger.warning(f"Tam duplikat: {n_exact}")
    if n_near: logger.warning(f"Yakin duplikat: {n_near}")
    return df


def detect_outliers(df: pd.DataFrame,
                    col: str = "word_count",
                    method: str = "iqr") -> pd.DataFrame:
    """IQR yöntemiyle kelime sayısı aykırı değerlerini işaretle."""
    df = df.copy()
    if method == "iqr":
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[f"{col}_outlier"] = ~df[col].between(lower, upper)
        n = df[f"{col}_outlier"].sum()
        logger.info(f"Outlier ({col}, IQR): {n} adet | sınır=[{lower:.0f}, {upper:.0f}]")
    return df


def compute_quality_scores(df: pd.DataFrame,
                            text_col: str = "text") -> pd.DataFrame:
    """Her metin için kalite skoru ekle (0–1 arası)."""
    df = df.copy()

    # Alfanümerik karakter oranı
    def alpha_ratio(t):
        alnum = sum(c.isalnum() for c in t)
        return alnum / len(t) if t else 0

    # Ortalama cümle uzunluğu (proxy: nokta sayısına göre)
    def avg_sentence_len(t):
        sentences = re.split(r"[.!?]", t)
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
        return np.mean(lengths) if lengths else 0

    df["alpha_ratio"]    = df[text_col].apply(alpha_ratio)
    df["avg_sent_len"]   = df[text_col].apply(avg_sentence_len)

    # Kalite skoru: alfanümerik oran * min(1, kelime_say/1000)
    df["quality_score"] = (
        df["alpha_ratio"] * np.minimum(1.0, df["word_count"] / 1_000)
    ).round(4)

    return df


def clean_corpus(df: pd.DataFrame,
                 text_col: str = "text",
                 remove_dups: bool = True,
                 min_words: int = 1_000) -> pd.DataFrame:
    """
    Tam temizleme pipeline'ı:
    1) Gutenberg artefakt temizliği
    2) Whitespace normalizasyonu
    3) Duplikat tespiti
    4) Çok kısa metin filtresi
    5) Kelime sayılarını güncelle
    6) Kalite skoru hesapla
    """
    logger.info(f"Temizleme basladi: {len(df)} eser")

    # 1-2: Metin temizliği
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text_pipeline)

    # Güncel kelime/karakter sayıları
    df["word_count"] = df[text_col].str.split().str.len()
    df["char_count"] = df[text_col].str.len()

    # 3: Duplikat tespiti (uyarı, silme yok — küçük corpus)
    df = detect_duplicates(df, text_col)

    # 4: Çok kısa metinleri filtrele
    n_before = len(df)
    df = df[df["word_count"] >= min_words].copy()
    n_removed = n_before - len(df)
    if n_removed:
        logger.warning(f"{n_removed} eser kalite filtresiyle cikti (< {min_words} kelime)")

    # 5: Outlier tespiti (yalnız bilgi amaçlı)
    df = detect_outliers(df, "word_count")

    # 6: Kalite skoru
    df = compute_quality_scores(df, text_col)

    logger.info(f"Temizleme tamamlandi: {len(df)} eser kaldi")
    return df


def save_clean_corpus(df: pd.DataFrame,
                      output_dir: str = "data/processed") -> str:
    """Temizlenmiş corpus'u parquet olarak kaydet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "corpus_clean.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Temiz corpus kaydedildi: {path}")
    return str(path)
