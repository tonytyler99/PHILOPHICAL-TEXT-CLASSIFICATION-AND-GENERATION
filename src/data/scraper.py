"""
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
                text = text[text.index("\n") + 1:]
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
