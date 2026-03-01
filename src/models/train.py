"""
PHIL-TEXT: Model Egitim Modulu
Geleneksel ML (SVM, RF) + Transformer fine-tuning.
"""
import joblib, json
from pathlib import Path
from loguru import logger
import numpy as np
from sklearn.svm import SVC, LinearSVC
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
    "linearsvc": lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
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
        meta_path = str(Path(filepath).with_suffix("")) + "_meta.json"
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
