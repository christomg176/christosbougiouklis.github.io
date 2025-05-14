import json
import os
from utils import simple_summarize, retrieve_matching_summary

STORE_PATH = "document_store.json"


def ingest_document(doc_text):
    if "concert" not in doc_text.lower() and "tour" not in doc_text.lower():
        return "Sorry, I cannot ingest documents with other themes."

    summary = simple_summarize(doc_text)
    db = []

    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, 'r') as f:
            db = json.load(f)

    db.append(summary)
    with open(STORE_PATH, 'w') as f:
        json.dump(db, f)

    return summary


def answer_query(user_query):
    if not os.path.exists(STORE_PATH):
        return "No documents have been ingested yet."

    with open(STORE_PATH, 'r') as f:
        db = json.load(f)

    return retrieve_matching_summary(user_query, db)
