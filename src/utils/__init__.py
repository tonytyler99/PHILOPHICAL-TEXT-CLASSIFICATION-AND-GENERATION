"""Yardimci fonksiyonlar"""
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
