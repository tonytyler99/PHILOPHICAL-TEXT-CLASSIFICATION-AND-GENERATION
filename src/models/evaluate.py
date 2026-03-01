"""PHIL-TEXT: Model Degerlendirme"""
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

def evaluate_transformer(model, tokenizer, texts, labels, id2label=None,
                          batch_size=32, max_length=512):
    import torch
    device = next(model.parameters()).device
    texts = list(texts)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
    all_preds = np.array(all_preds)
    metrics = evaluate_classification(np.array(labels), all_preds, id2label)
    metrics['predictions'] = all_preds.tolist()
    return metrics

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
