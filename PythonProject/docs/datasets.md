# 📚 Datasets Used

---

## 🧊 3D Models

- **Source**: [Sketchfab](https://sketchfab.com)
- **Query Examples**: `molecule`, `DNA`, `structure`
- **Sample Models**: Water molecule, DNA helix
- **Ingested via**: Sketchfab API
- **License**: Creative Commons Attribution (CC-BY)
- **Usage**: Cached in `cache.db`, displayed on frontend

---

## 🧠 Text Corpora

- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets)
- **Example**: GPT-2 pretraining data, QA datasets (e.g. SQuAD)
- **License**: CC-BY / MIT (varies)
- **Usage**: Input for NLP models (Q&A, summarization, fine-tuning)

---

## 📊 Logs & Validation

- Logs written to: `logs/etl.log`
- Catalog JSON: `static/data/catalog.json`
