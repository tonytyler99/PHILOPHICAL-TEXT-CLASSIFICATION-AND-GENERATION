"""
PHIL-TEXT: Kaynak Dosya Kurulum Scripti
C:\PHIL-TEXT klasöründe çalıştırın: python setup_phil_text.py
"""
import os
from pathlib import Path

BASE = Path(".")

FILES = {}

# ── src/__init__.py ──
FILES["src/__init__.py"] = '''"""PHIL-TEXT: Felsefe Metinleri Analiz ve Uretim Sistemi"""
__version__ = "0.1.0"
'''

# ── src/data/__init__.py ──
FILES["src/data/__init__.py"] = '''"""Veri yukleme ve on isleme modulleri"""
from src.data.load_data import load_data, load_corpus, get_data_summary
from src.data.preprocess import TextPreprocessor
from src.data.scraper import PhilosophyScraper
'''

# ── src/data/load_data.py ──
FILES["src/data/load_data.py"] = '''"""
PHIL-TEXT: Veri Yukleme Modulu
Felsefe metinlerini cesitli kaynaklardan yukler ve birlestirir.
"""
import pandas as pd
import json
import glob
from pathlib import Path
from loguru import logger

PHILOSOPHERS = {
    "platon": {"era": "antik_yunan", "school": "idealizm"},
    "aristoteles": {"era": "antik_yunan", "school": "realizm"},
    "epikuros": {"era": "antik_yunan", "school": "epikurosculuk"},
    "marcus_aurelius": {"era": "antik_yunan", "school": "stoacilik"},
    "seneca": {"era": "antik_yunan", "school": "stoacilik"},
    "thomas_aquinas": {"era": "orta_cag", "school": "skolastisizm"},
    "augustinus": {"era": "orta_cag", "school": "skolastisizm"},
    "descartes": {"era": "modern", "school": "rasyonalizm"},
    "spinoza": {"era": "modern", "school": "rasyonalizm"},
    "leibniz": {"era": "modern", "school": "rasyonalizm"},
    "locke": {"era": "modern", "school": "ampirizm"},
    "hume": {"era": "modern", "school": "ampirizm"},
    "kant": {"era": "modern", "school": "transandantal_idealizm"},
    "hegel": {"era": "modern", "school": "idealizm"},
    "schopenhauer": {"era": "modern", "school": "pesimizm"},
    "nietzsche": {"era": "cagdas", "school": "varoluşculuk"},
    "heidegger": {"era": "cagdas", "school": "varoluşculuk"},
    "sartre": {"era": "cagdas", "school": "varoluşculuk"},
    "camus": {"era": "cagdas", "school": "absurdizm"},
    "wittgenstein": {"era": "cagdas", "school": "analitik_felsefe"},
    "russell": {"era": "cagdas", "school": "analitik_felsefe"},
    "foucault": {"era": "cagdas", "school": "postmodernizm"},
    "derrida": {"era": "cagdas", "school": "postmodernizm"},
}

ERAS = ["antik_yunan", "orta_cag", "modern", "cagdas"]
SCHOOLS = [
    "stoacilik", "varoluşculuk", "rasyonalizm", "ampirizm",
    "idealizm", "realizm", "analitik_felsefe", "postmodernizm",
    "skolastisizm", "transandantal_idealizm", "pesimizm",
    "absurdizm", "epikurosculuk",
]


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {filepath}")
    logger.info(f"CSV yukleniyor: {filepath}")
    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
    return df


def load_json(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"JSON yuklendi: {filepath} ({len(data)} kayit)")
    return data


def load_corpus(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Tum felsefe metinlerini yukle.
    Beklenen yapi: data/raw/filozof_adi/eser_adi.txt
    """
    records = []
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Veri klasoru bulunamadi: {data_dir}")
        return pd.DataFrame()

    for philosopher_dir in sorted(data_path.iterdir()):
        if not philosopher_dir.is_dir():
            continue
        philosopher = philosopher_dir.name.lower()
        meta = PHILOSOPHERS.get(philosopher, {"era": "bilinmiyor", "school": "bilinmiyor"})
        for txt_file in sorted(philosopher_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            records.append({
                "philosopher": philosopher, "work": txt_file.stem,
                "text": text, "era": meta["era"], "school": meta["school"],
                "char_count": len(text), "word_count": len(text.split()),
                "source_file": str(txt_file),
            })

    df = pd.DataFrame(records)
    logger.info(f"Corpus yuklendi: {len(df)} eser, {df['philosopher'].nunique()} filozof")
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
        raise ValueError(f"Desteklenmeyen format: {path.suffix}")
    return loader(filepath, **kwargs)


def chunk_texts(df: pd.DataFrame, chunk_size: int = 512, overlap: int = 64) -> pd.DataFrame:
    """Uzun metinleri egitim icin parcalara bol."""
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
    logger.info(f"Metin parcalama: {len(df)} eser -> {len(result)} parca")
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
'''

# ── src/data/preprocess.py ──
FILES["src/data/preprocess.py"] = '''"""
PHIL-TEXT: Metin On Isleme Modulu
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except ImportError:
    pass


class TextPreprocessor:
    def __init__(self, language="english", use_spacy=False):
        self.language = language
        self.use_spacy = use_spacy and nlp is not None
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set()

    def clean_text(self, text):
        text = re.sub(r"http\\S+|www\\S+", "", text)
        text = re.sub(r"\\[.*?\\]", "", text)
        text = re.sub(r"\\n+", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def normalize_text(self, text):
        return text.lower()

    def remove_stopwords(self, text):
        words = text.split()
        return " ".join(w for w in words if w.lower() not in self.stop_words)

    def lemmatize(self, text):
        if self.use_spacy:
            doc = nlp(text)
            return " ".join(token.lemma_ for token in doc if not token.is_space)
        words = text.split()
        return " ".join(self.lemmatizer.lemmatize(w) for w in words)

    def preprocess(self, text, steps=None):
        if steps is None:
            steps = ["clean", "normalize", "remove_stopwords", "lemmatize"]
        pipeline = {"clean": self.clean_text, "normalize": self.normalize_text,
                     "remove_stopwords": self.remove_stopwords, "lemmatize": self.lemmatize}
        for step in steps:
            if step in pipeline:
                text = pipeline[step](text)
        return text

    def preprocess_dataframe(self, df, text_col="text", steps=None):
        df = df.copy()
        logger.info(f"Preprocessing basliyor: {len(df)} metin...")
        df[f"{text_col}_processed"] = df[text_col].apply(lambda x: self.preprocess(str(x), steps))
        logger.info("Preprocessing tamamlandi.")
        return df


def prepare_classification_data(df, text_col="text_processed", label_col="philosopher",
                                 test_size=0.2, val_size=0.1, random_state=42, min_samples=10):
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered = df[df[label_col].isin(valid_classes)].copy()

    labels = sorted(df_filtered[label_col].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df_filtered["label_id"] = df_filtered[label_col].map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        df_filtered[text_col].values, df_filtered["label_id"].values,
        test_size=test_size, random_state=random_state, stratify=df_filtered["label_id"].values)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)

    result = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
              "X_test": X_test, "y_test": y_test, "label2id": label2id,
              "id2label": id2label, "num_classes": len(labels)}
    logger.info(f"Veri bolme: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return result
'''

# ── src/data/scraper.py ──
FILES["src/data/scraper.py"] = '''"""
PHIL-TEXT: Felsefe Metni Toplama Modulu
Project Gutenberg'den metin indirme.
"""
import os, json, time
from pathlib import Path
from loguru import logger

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

GUTENBERG_BOOKS = {
    "platon": {"republic": 1497, "apology": 1656, "symposium": 1600, "phaedo": 1658, "meno": 1643},
    "aristoteles": {"nicomachean_ethics": 8438, "politics": 6762, "poetics": 1974, "metaphysics": 6763},
    "marcus_aurelius": {"meditations": 2680},
    "descartes": {"discourse_on_method": 59, "meditations": 59861},
    "kant": {"critique_of_pure_reason": 4280, "critique_of_practical_reason": 5683,
             "fundamental_principles": 5682},
    "nietzsche": {"thus_spake_zarathustra": 1998, "beyond_good_and_evil": 4363,
                  "genealogy_of_morals": 52319},
    "hume": {"enquiry_human_understanding": 9662, "treatise_human_nature": 4705},
    "locke": {"essay_human_understanding": 10615, "two_treatises_government": 7370},
    "schopenhauer": {"world_as_will_and_idea": 38427},
    "spinoza": {"ethics": 3800},
}


class PhilosophyScraper:
    def __init__(self, output_dir="data/raw", delay=2.0):
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_gutenberg(self, book_id):
        if not HAS_REQUESTS:
            raise ImportError("requests kutuphanesi gerekli: pip install requests")
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        logger.info(f"Indiriliyor: Gutenberg #{book_id}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        text = response.text
        for marker in ["*** START OF", "***START OF"]:
            if marker in text:
                text = text[text.index(marker):]
                text = text[text.index("\\n") + 1:]
                break
        for marker in ["*** END OF", "***END OF"]:
            if marker in text:
                text = text[:text.index(marker)]
                break
        return text.strip()

    def download_philosopher(self, philosopher):
        books = GUTENBERG_BOOKS.get(philosopher, {})
        if not books:
            logger.warning(f"Kayitli kitap yok: {philosopher}")
            return {}
        phil_dir = self.output_dir / philosopher
        phil_dir.mkdir(exist_ok=True)
        results = {}
        for work_name, book_id in books.items():
            filepath = phil_dir / f"{work_name}.txt"
            if filepath.exists():
                logger.info(f"Zaten mevcut: {filepath}")
                results[work_name] = str(filepath)
                continue
            try:
                text = self.fetch_gutenberg(book_id)
                filepath.write_text(text, encoding="utf-8")
                results[work_name] = str(filepath)
                logger.info(f"Kaydedildi: {filepath} ({len(text)} karakter)")
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Hata: {work_name} (#{book_id}): {e}")
                results[work_name] = None
        return results

    def download_all(self):
        all_results = {}
        total = sum(len(b) for b in GUTENBERG_BOOKS.values())
        logger.info(f"Toplam {total} eser indirilecek...")
        for philosopher in GUTENBERG_BOOKS:
            all_results[philosopher] = self.download_philosopher(philosopher)
        success = sum(1 for r in all_results.values() for v in r.values() if v)
        logger.info(f"Tamamlandi: {success}/{total} eser basariyla indirildi.")
        return all_results

    def get_download_status(self):
        import pandas as pd
        records = []
        for philosopher, books in GUTENBERG_BOOKS.items():
            for work, book_id in books.items():
                filepath = self.output_dir / philosopher / f"{work}.txt"
                exists = filepath.exists()
                size = filepath.stat().st_size if exists else 0
                records.append({"philosopher": philosopher, "work": work,
                                "gutenberg_id": book_id, "downloaded": exists,
                                "file_size_kb": round(size / 1024, 1)})
        return pd.DataFrame(records)
'''

# ── src/features/__init__.py ──
FILES["src/features/__init__.py"] = '''"""
PHIL-TEXT: Feature Engineering
TF-IDF, stilistik ozellikler, transformer embeddings.
"""
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TextFeatureBuilder:
    def __init__(self):
        self.tfidf_vectorizer = None

    def build_tfidf(self, texts, max_features=10000, ngram_range=(1, 2)):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, min_df=2, max_df=0.95)
        features = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF: {features.shape}")
        return features

    def transform_tfidf(self, texts):
        if self.tfidf_vectorizer is None:
            raise ValueError("Once build_tfidf cagirin.")
        return self.tfidf_vectorizer.transform(texts)

    def build_bow(self, texts, max_features=5000):
        self.count_vectorizer = CountVectorizer(max_features=max_features, min_df=2)
        features = self.count_vectorizer.fit_transform(texts)
        logger.info(f"BoW: {features.shape}")
        return features

    @staticmethod
    def build_stylistic_features(df, text_col="text"):
        df = df.copy()
        texts = df[text_col].astype(str)
        df["word_count"] = texts.str.split().str.len()
        df["char_count"] = texts.str.len()
        df["sentence_count"] = texts.str.count(r"[.!?]+")
        df["avg_word_length"] = texts.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
        df["avg_sentence_length"] = df["word_count"] / df["sentence_count"].replace(0, 1)
        df["unique_word_ratio"] = texts.apply(
            lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))
        df["comma_ratio"] = texts.str.count(",") / df["word_count"].replace(0, 1)
        df["question_ratio"] = texts.str.count(r"\\?") / df["sentence_count"].replace(0, 1)
        df["long_word_ratio"] = texts.apply(
            lambda x: sum(1 for w in x.split() if len(w) > 8) / max(len(x.split()), 1))
        logger.info("Stilistik ozellikler eklendi")
        return df

    @staticmethod
    def build_transformer_embeddings(texts, model_name="bert-base-uncased",
                                      batch_size=16, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Transformer embeddings: {model_name} ({device})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        result = np.vstack(all_embeddings)
        logger.info(f"Embedding shape: {result.shape}")
        return result
'''

# ── src/models/__init__.py ──
FILES["src/models/__init__.py"] = '''"""Model egitim, tahmin ve degerlendirme modulleri"""
'''

# ── src/models/train.py ──
FILES["src/models/train.py"] = '''"""
PHIL-TEXT: Model Egitim Modulu
Geleneksel ML (SVM, RF) + Transformer fine-tuning.
"""
import joblib, json
from pathlib import Path
from loguru import logger
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

TRADITIONAL_MODELS = {
    "svm": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "svm_linear": lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1),
}


def create_tfidf_pipeline(model_name="svm", max_features=10000):
    if model_name not in TRADITIONAL_MODELS:
        raise ValueError(f"Bilinmeyen model: {model_name}")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", TRADITIONAL_MODELS[model_name]()),
    ])
    logger.info(f"Pipeline: TF-IDF({max_features}) -> {model_name}")
    return pipeline


def train_traditional(pipeline, X_train, y_train):
    logger.info(f"Egitim basliyor: {len(X_train)} ornek...")
    pipeline.fit(X_train, y_train)
    logger.info("Egitim tamamlandi.")
    return pipeline


def train_transformer(train_texts, train_labels, val_texts, val_labels,
                      model_name="bert-base-uncased", num_labels=10,
                      epochs=5, batch_size=16, learning_rate=2e-5,
                      max_length=512, output_dir="models/saved/transformer"):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Transformer egitim: {model_name} | {device} | {epochs} epoch")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels).to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True,
                                        max_length=max_length, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_loader = DataLoader(TextDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_texts, val_labels), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_transformer(model, tokenizer, output_dir)

    return {"best_val_accuracy": best_val_acc, "history": history}


def save_model(model, filepath, metadata=None):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    if metadata:
        meta_path = filepath.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as f: json.dump(metadata, f, indent=2)
    logger.info(f"Model kaydedildi: {filepath}")

def load_model(filepath):
    model = joblib.load(filepath)
    logger.info(f"Model yuklendi: {filepath}")
    return model

def save_transformer(model, tokenizer, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_transformer(output_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    return model, tokenizer
'''

# ── src/models/evaluate.py ──
FILES["src/models/evaluate.py"] = '''"""PHIL-TEXT: Model Degerlendirme"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from loguru import logger
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score

def evaluate_classification(y_true, y_pred, id2label=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    target_names = [id2label[i] for i in sorted(id2label.keys())] if id2label else None
    metrics["report"] = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}")
    return metrics

def plot_confusion_matrix(y_true, y_pred, id2label=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    labels = [id2label[i] for i in sorted(id2label.keys())] if id2label else None
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
    ax.set_title("PHIL-TEXT: Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return cm

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    logger.info(f"CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return {"cv_scores": scores.tolist(), "mean_f1": scores.mean(), "std_f1": scores.std()}

def compare_models(results, save_path=None):
    df = pd.DataFrame([{"model": n, **{k: v for k, v in m.items() if isinstance(v, (int, float))}}
                        for n, m in results.items()])
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.melt(id_vars=["model"], value_vars=["accuracy", "f1_weighted", "f1_macro"],
                      var_name="Metrik", value_name="Skor")
    sns.barplot(data=df_plot, x="model", y="Skor", hue="Metrik", ax=ax)
    ax.set_title("Model Karsilastirma", fontweight="bold"); ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return df
'''

# ── src/models/predict.py ──
FILES["src/models/predict.py"] = '''"""PHIL-TEXT: Tahmin ve Metin Uretim Modulu"""
import numpy as np
from loguru import logger

def predict_philosopher(model, texts, id2label=None):
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    results = []
    for i, text in enumerate(texts):
        result = {"text_preview": text[:100] + "...", "predicted_label": int(predictions[i]),
                  "predicted_name": id2label[predictions[i]] if id2label else str(predictions[i])}
        if probabilities is not None:
            top_3 = probabilities[i].argsort()[-3:][::-1]
            result["top_3"] = [{"label": id2label[idx] if id2label else str(idx),
                                "probability": round(float(probabilities[i][idx]), 4)} for idx in top_3]
        results.append(result)
    logger.info(f"{len(texts)} metin siniflandirildi")
    return results

def predict_transformer(model, tokenizer, texts, id2label=None, max_length=512):
    import torch
    device = next(model.parameters()).device
    model.eval()
    results = []
    for text in texts:
        encoded = tokenizer(text, truncation=True, padding=True,
                            max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred_id = int(probs.argmax())
        top_3 = probs.argsort()[-3:][::-1]
        results.append({"text_preview": text[:100] + "...", "predicted_name": id2label[pred_id] if id2label else str(pred_id),
                        "confidence": round(float(probs[pred_id]), 4),
                        "top_3": [{"label": id2label[int(i)] if id2label else str(int(i)),
                                   "probability": round(float(probs[i]), 4)} for i in top_3]})
    return results

def generate_text(prompt, model_dir="models/saved/generator", max_length=300,
                  temperature=0.8, top_p=0.92, num_return=1):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=max_length, temperature=temperature,
                              top_p=top_p, num_return_sequences=num_return, do_sample=True,
                              pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
'''

# ── src/utils/__init__.py ──
FILES["src/utils/__init__.py"] = '''"""Yardimci fonksiyonlar"""
import yaml, json, sys
from pathlib import Path
from loguru import logger

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logger(log_file="logs/app.log", level="INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", retention="30 days", level=level)
    return logger

def save_json(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
'''

# ── api/app.py ──
FILES["api/app.py"] = '''"""PHIL-TEXT: FastAPI Model Serving"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="PHIL-TEXT API", version="0.1.0")
classifier = None
id2label = None

class ClassifyRequest(BaseModel):
    text: str
    top_k: int = 3

class ClassifyResponse(BaseModel):
    predicted_philosopher: str
    confidence: float
    top_predictions: list

@app.on_event("startup")
def load_models():
    global classifier, id2label
    model_path = Path("models/saved/classifier.pkl")
    if model_path.exists():
        classifier = joblib.load(model_path)
    labels_path = Path("models/saved/id2label.json")
    if labels_path.exists():
        import json
        with open(labels_path) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

@app.get("/")
def root():
    return {"service": "PHIL-TEXT API", "status": "active"}

@app.get("/health")
def health():
    return {"status": "healthy", "classifier_loaded": classifier is not None}

@app.post("/classify", response_model=ClassifyResponse)
def classify_text(request: ClassifyRequest):
    if classifier is None:
        raise HTTPException(503, "Model yuklenmedi")
    try:
        probs = classifier.predict_proba([request.text])[0]
        pred_id = int(probs.argmax())
        top_k = probs.argsort()[-request.top_k:][::-1]
        return ClassifyResponse(
            predicted_philosopher=id2label.get(pred_id, str(pred_id)),
            confidence=round(float(probs[pred_id]), 4),
            top_predictions=[{"philosopher": id2label.get(int(i), str(int(i))),
                              "probability": round(float(probs[i]), 4)} for i in top_k])
    except Exception as e:
        raise HTTPException(400, str(e))
'''

# ── tests/__init__.py ──
FILES["tests/__init__.py"] = ''

# ── tests/test_phil_text.py ──
FILES["tests/test_phil_text.py"] = '''"""PHIL-TEXT Test Suite"""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_corpus():
    return pd.DataFrame({
        "philosopher": ["platon", "platon", "nietzsche", "nietzsche", "kant", "kant"],
        "work": ["republic", "republic", "zarathustra", "zarathustra", "critique", "critique"],
        "text": [
            "The myth of the cave illustrates the nature of reality and knowledge.",
            "Justice is the harmony of the soul when each part fulfills its function.",
            "God is dead and we have killed him. What festivals of atonement shall we invent?",
            "Man is a rope stretched between animal and overman a rope over an abyss.",
            "All our knowledge begins with the senses proceeds to understanding and ends with reason.",
            "Act only according to that maxim whereby you can will that it become a universal law.",
        ],
        "era": ["antik_yunan", "antik_yunan", "cagdas", "cagdas", "modern", "modern"],
        "school": ["idealizm", "idealizm", "varoluşculuk", "varoluşculuk",
                    "transandantal_idealizm", "transandantal_idealizm"],
    })

class TestPreprocessor:
    def test_clean_text(self):
        from src.data.preprocess import TextPreprocessor
        prep = TextPreprocessor()
        result = prep.clean_text("Hello  world\\n\\nnew line   extra spaces")
        assert "\\n" not in result

    def test_normalize(self):
        from src.data.preprocess import TextPreprocessor
        prep = TextPreprocessor()
        assert prep.normalize_text("Hello WORLD") == "hello world"

    def test_preprocess_df(self, sample_corpus):
        from src.data.preprocess import TextPreprocessor
        prep = TextPreprocessor()
        result = prep.preprocess_dataframe(sample_corpus, steps=["clean", "normalize"])
        assert "text_processed" in result.columns

class TestDataLoading:
    def test_philosophers_dict(self):
        from src.data.load_data import PHILOSOPHERS
        assert "platon" in PHILOSOPHERS
        assert "nietzsche" in PHILOSOPHERS

    def test_get_summary(self, sample_corpus):
        from src.data.load_data import get_data_summary
        summary = get_data_summary(sample_corpus)
        assert summary["total_records"] == 6

class TestFeatures:
    def test_tfidf(self, sample_corpus):
        from src.features import TextFeatureBuilder
        builder = TextFeatureBuilder()
        features = builder.build_tfidf(sample_corpus["text"].values, max_features=100)
        assert features.shape[0] == 6

    def test_stylistic(self, sample_corpus):
        from src.features import TextFeatureBuilder
        result = TextFeatureBuilder.build_stylistic_features(sample_corpus)
        assert "word_count" in result.columns

class TestModel:
    def test_pipeline_creation(self):
        from src.models.train import create_tfidf_pipeline
        pipeline = create_tfidf_pipeline("svm", max_features=100)
        assert pipeline is not None
'''

# ── docker/Dockerfile ──
FILES["docker/Dockerfile"] = '''FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# ── docker/docker-compose.yml ──
FILES["docker/docker-compose.yml"] = '''version: "3.8"
services:
  phil-text-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
      - ../configs:/app/configs
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000
    restart: unless-stopped
'''

# ── README.md ──
FILES["README.md"] = '''# PHIL-TEXT: Felsefe Metinleri Analiz ve Uretim Sistemi

Felsefe metinlerini **siniflandiran** ve belirli filozoflarin uslubuyla **yeni metinler ureten** AI/ML projesi.

## Ozellikler
- Metin Siniflandirma: Filozof, akim, donem tahmini
- Metin Uretimi: GPT-2 fine-tuning ile filozof tarzi uretim
- Geleneksel ML: TF-IDF + SVM/RF
- Derin Ogrenme: BERT/RoBERTa fine-tuning (GPU destekli)
- REST API: FastAPI ile model serving

## Hizli Baslangic
```bash
cd C:\\PHIL-TEXT
venv\\Scripts\\activate
python -c "from src.data.scraper import PhilosophyScraper; PhilosophyScraper().download_all()"
pytest tests/ -v
uvicorn api.app:app --reload --port 8000
```
'''

# ═══════════════════════════════════════
#  Dosyalari olustur
# ═══════════════════════════════════════
created = 0
for filepath, content in FILES.items():
    full_path = BASE / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    created += 1
    print(f"  ✅ {filepath}")

# .gitkeep dosyalari
gitkeep_dirs = ["data/raw", "data/processed", "data/external", "models/saved", "logs"]
for d in gitkeep_dirs:
    gk = BASE / d / ".gitkeep"
    gk.parent.mkdir(parents=True, exist_ok=True)
    gk.touch()
    print(f"  ✅ {d}/.gitkeep")

print(f"\n🎉 Toplam {created} dosya + {len(gitkeep_dirs)} .gitkeep oluşturuldu!")
print("Simdi test etmek icin: pytest tests/ -v")
