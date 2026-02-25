"""
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
        df["question_ratio"] = texts.str.count(r"\?") / df["sentence_count"].replace(0, 1)
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
