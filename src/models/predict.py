"""PHIL-TEXT: Tahmin ve Metin Uretim Modulu"""
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
