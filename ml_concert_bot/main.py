from rag_engine import ingest_document, answer_query

# Ingesting a sample concert document
doc_text = """
Lady Gaga will tour in Autumn 2025 visiting Paris, Berlin, and Milan. Special guest: Elton John.
"""
summary = ingest_document(doc_text)
print("Ingested Summary:", summary)

# Asking a question
query = "Where is Lady Gaga going in Autumn 2025?"
answer = answer_query(query)
print("Answer:", answer)
