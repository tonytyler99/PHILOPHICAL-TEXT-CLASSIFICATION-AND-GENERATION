# PHIL-TEXT: Felsefe Metinleri Analiz ve Uretim Sistemi

Felsefe metinlerini **siniflandiran** ve belirli filozoflarin uslubuyla **yeni metinler ureten** AI/ML projesi.

## Ozellikler
- Metin Siniflandirma: Filozof, akim, donem tahmini
- Metin Uretimi: GPT-2 fine-tuning ile filozof tarzi uretim
- Geleneksel ML: TF-IDF + SVM/RF
- Derin Ogrenme: BERT/RoBERTa fine-tuning (GPU destekli)
- REST API: FastAPI ile model serving

## Hizli Baslangic
```bash
cd C:\PHIL-TEXT
venv\Scripts\activate
python -c "from src.data.scraper import PhilosophyScraper; PhilosophyScraper().download_all()"
pytest tests/ -v
uvicorn api.app:app --reload --port 8000
```
