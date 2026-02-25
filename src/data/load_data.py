"""
PHIL-TEXT: Data Loading Module
Loads and combines philosophy texts from various sources.
"""
import pandas as pd
import json
import glob
from pathlib import Path
from loguru import logger

PHILOSOPHERS = {
    "platon": {"era": "ancient_greek", "school": "idealism"},
    "aristoteles": {"era": "ancient_greek", "school": "realism"},
    "epikuros": {"era": "ancient_greek", "school": "epicureanism"},
    "marcus_aurelius": {"era": "ancient_greek", "school": "stoicism"},
    "seneca": {"era": "ancient_greek", "school": "stoicism"},
    "thomas_aquinas": {"era": "middle-age", "school": "scholasticism"},
    "augustinus": {"era": "middle-age", "school": "scholasticism"},
    "descartes": {"era": "modern", "school": "rationalism"},
    "spinoza": {"era": "modern", "school": "rationalism"},
    "leibniz": {"era": "modern", "school": "rationalism"},
    "locke": {"era": "modern", "school": "empiricism"},
    "hume": {"era": "modern", "school": "empiricism"},
    "kant": {"era": "modern", "school": "transandantal_idealism"},
    "hegel": {"era": "modern", "school": "idealism"},
    "schopenhauer": {"era": "modern", "school": "pessimism"},
    "nietzsche": {"era": "Contemporary", "school": "existentialism"},
    "heidegger": {"era": "Contemporary", "school": "existentialism"},
    "sartre": {"era": "Contemporary", "school": "existentialism"},
    "camus": {"era": "Contemporary", "school": "absurdism"},
    "wittgenstein": {"era": "Contemporary", "school": "analytical_philosophy"},
    "russell": {"era": "Contemporary", "school": "analytical_philosophy"},
    "foucault": {"era": "Contemporary", "school": "postmodernism"},
    "derrida": {"era": "Contemporary", "school": "postmodernism"},
}

ERAS = ["ancient_greek", "middle-age", "modern", "Contemporary"]
SCHOOLS = [
    "stoicism", "existentialism", "rationalism", "empiricism",
    "idealism", "realism", "analytical_philosophy", "postmodernism",
    "scholasticism", "transandantal_idealism", "pessimism",
    "absurdism", "epicureanism",
]


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Folder not found : {filepath}")
    logger.info(f"CSV is uploading: {filepath}")
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Uploaded: {df.shape[0]} Row, {df.shape[1]} columns")
    return df


def load_json(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"JSON is uploaded: {filepath} ({len(data)} records)")
    return data


def load_corpus(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Tum felsefe metinlerini yukle.
    Beklenen yapi: data/raw/filozof_adi/eser_adi.txt
    """
    records = []
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return pd.DataFrame()

    for philosopher_dir in sorted(data_path.iterdir()):
        if not philosopher_dir.is_dir():
            continue
        philosopher = philosopher_dir.name.lower()
        meta = PHILOSOPHERS.get(philosopher, {"era": "unknown", "school": "unknown"})
        for txt_file in sorted(philosopher_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            records.append({
                "philosopher": philosopher, "work": txt_file.stem,
                "text": text, "era": meta["era"], "school": meta["school"],
                "char_count": len(text), "word_count": len(text.split()),
                "source_file": str(txt_file),
            })

    df = pd.DataFrame(records)
    logger.info(f"Corpus uploaded: {len(df)} works, {df['philosopher'].nunique()} philosophers")
    return df


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    path = Path(filepath)
    loaders = {
        ".csv": load_csv,
        ".json": lambda f, **kw: pd.DataFrame(load_json(f)),
        ".xlsx": lambda f, **kw: pd.read_excel(f, **kw),
        ".parquet": lambda f, **kw: pd.read_parquet(f, **kw),
    }
    loader = loaders.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return loader(filepath, **kwargs)


def chunk_texts(df: pd.DataFrame, chunk_size: int = 512, overlap: int = 64) -> pd.DataFrame:
    """Break down long texts into smaller parts for educational purposes."""
    chunks = []
    for _, row in df.iterrows():
        words = row["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:
                continue
            chunks.append({
                "philosopher": row["philosopher"], "work": row["work"],
                "era": row["era"], "school": row["school"],
                "text": " ".join(chunk_words), "chunk_idx": i // (chunk_size - overlap),
            })
    result = pd.DataFrame(chunks)
    logger.info(f"splitting: {len(df)} works -> {len(result)} chunks")
    return result


def get_data_summary(df: pd.DataFrame) -> dict:
    summary = {"total_records": len(df), "columns": list(df.columns),
               "missing_values": df.isnull().sum().to_dict()}
    if "philosopher" in df.columns:
        summary["philosophers"] = df["philosopher"].value_counts().to_dict()
    if "era" in df.columns:
        summary["eras"] = df["era"].value_counts().to_dict()
    if "school" in df.columns:
        summary["schools"] = df["school"].value_counts().to_dict()
    if "text" in df.columns:
        summary["avg_text_length"] = df["text"].str.len().mean()
        summary["total_words"] = df["text"].str.split().str.len().sum()
    return summary
