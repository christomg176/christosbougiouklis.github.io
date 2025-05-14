import serpapi

SERP_API_KEY = "your_api_key_here"

def search_artist_tour(artist_name):
    query = f"{artist_name} 2025 concert tour"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 5,
    }

    search = serpapi.search(params)
    results = search.get_dict()

    snippets = []
    for r in results.get("organic_results", []):
        if "snippet" in r:
            snippets.append(r["snippet"])

    return "\n".join(snippets[:3]) if snippets else "No tour information found."
