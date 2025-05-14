def simple_summarize(text):
    # Simulated summary (real model: use OpenAI, Hugging Face, etc.)
    return text.strip().split('.')[0] + "."

def retrieve_matching_summary(query, summaries):
    query_lower = query.lower()
    for summary in summaries:
        if any(word in summary.lower() for word in query_lower.split()):
            return summary
    return "Sorry, I couldn’t find relevant info in the ingested documents."
