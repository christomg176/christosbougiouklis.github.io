import pandas as pd
import requests
import sqlite3
import os
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("static/data")
LOG_FILE = Path("logs/etl.log")
DB_FILE = Path("cache.db")
CATALOG_FILE = Path("static/data/catalog.json")

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} - {message}\n")

def ingest_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(url, names=cols)
    df.dropna(inplace=True)
    df.to_csv(DATA_DIR / "iris_clean.csv", index=False)
    log("Ingested Iris dataset from UCI ML repo.")

def ingest_sketchfab_models(query="molecule", license="cc-by"):
    url = f"https://api.sketchfab.com/v3/models?license={license}&q={query}&downloadable=true"
    response = requests.get(url)
    if response.status_code != 200:
        log("Sketchfab API failed")
        return

    models = response.json().get("results", [])
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT,
            url TEXT,
            license TEXT,
            download TEXT
        )
    """)
    added = 0
    for model in models:
        try:
            cursor.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?)", (
                model["uid"],
                model["name"],
                model["viewerUrl"],
                model.get("license", {}).get("label", "Unknown"),
                model.get("downloadUrl", "N/A")
            ))
            added += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    log(f"Cached {added} Sketchfab models with query '{query}'.")

def generate_catalog():
    catalog = [
        {
            "name": "Iris Dataset",
            "source": "UCI ML Repository",
            "domain": "Scientific",
            "license": "CC BY 4.0",
            "path": str(DATA_DIR / "iris_clean.csv")
        },
        {
            "name": "Sketchfab Molecules",
            "source": "Sketchfab API",
            "domain": "3D Models",
            "license": "CC BY",
            "cache": str(DB_FILE)
        }
    ]
    import json
    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2)
    log("Generated dataset catalog.")

if __name__ == "__main__":
    ingest_iris()
    ingest_sketchfab_models()
    generate_catalog()
