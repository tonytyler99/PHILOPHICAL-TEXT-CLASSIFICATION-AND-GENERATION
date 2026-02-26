"""Data module for PHIL-TEXT: loading and processing philosophy texts."""
from src.data.load_data import load_data, load_corpus, chunk_texts, get_data_summary
from src.data.preprocess import TextPreprocessor, prepare_classification_data
from src.data.clean_data import (
    clean_corpus,
    clean_text_pipeline,
    remove_gutenberg_artifacts,
    detect_duplicates,
    detect_outliers,
    compute_quality_scores,
    save_clean_corpus,
)
from src.data.scraper import PhilosophyScraper

__all__ = [
    "load_data",
    "load_corpus",
    "chunk_texts",
    "get_data_summary",
    "TextPreprocessor",
    "prepare_classification_data",
    "clean_corpus",
    "clean_text_pipeline",
    "remove_gutenberg_artifacts",
    "detect_duplicates",
    "detect_outliers",
    "compute_quality_scores",
    "save_clean_corpus",
    "PhilosophyScraper",
]

