"""PHIL-TEXT Test Suite"""
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
        result = prep.clean_text("Hello  world\n\nnew line   extra spaces")
        assert "\n" not in result

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
