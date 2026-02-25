"""
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
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
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
