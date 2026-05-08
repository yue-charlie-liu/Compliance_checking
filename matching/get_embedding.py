import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import sys
import hashlib
from datetime import datetime
from requests.exceptions import ConnectionError, Timeout, RequestException

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path("../chunking/flattened")  
OUTPUT_DIR = Path("./embedding")           
MODEL = "text-embedding-3-large"
CLIP_LENGTH = 30_000                       # safety clip length
FLATTEN_VERSION = "v1"                     # track flattening version

# -----------------------
# API Setup
# -----------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.api_keys import get_openai_api_key

client = OpenAI(api_key=get_openai_api_key())

# -----------------------
# Embedding Genetation
# -----------------------
def hash_id(text: str):
    """Stable hash ID for unique identification"""
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text: str, retries: int = 3):
    """Call OpenAI to get embedding with retry logic for connection errors"""
    text = text[:CLIP_LENGTH]
    
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=MODEL,
                input=text
            )
            return response.data[0].embedding
        except (ConnectionError, Timeout) as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⚠️ Connection error (attempt {attempt + 1}/{retries}): {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            raise

def load_existing(path: Path):
    """Load existing embeddings and index by title"""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    index = {item["title"]: item for item in data if item.get("title")}
    return index

# -----------------------
# Main Processing
# -----------------------
def process_all_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in INPUT_DIR.iterdir() if f.suffix == ".json")

    print(f"Found {len(files)} flatten json files")
    
    connection_errors = []
    api_errors = []

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name
        print(f"\nProcessing: {input_path.name}")

        # load new flatten data
        with open(input_path, encoding="utf-8") as f:
            new_data = json.load(f)

        # load existing embeddings
        existing_index = load_existing(output_path)

        updated_data = []

        for item in tqdm(new_data, desc="Items"):
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()

            if not title:
                continue

            regenerate = False

            # new item
            if title not in existing_index:
                regenerate = True
            else:
                old_item = existing_index[title]
                old_text = old_item.get("text", "").strip()
                # text changed
                if old_text != text:
                    regenerate = True
                # previously null embedding
                elif not old_item.get("embedding"):
                    regenerate = True

            if regenerate and text:
                try:
                    emb = get_embedding(text)
                except (ConnectionError, Timeout) as e:
                    connection_errors.append({
                        "title": title,
                        "error": f"{type(e).__name__}: {str(e)[:100]}"
                    })
                    print(f"❌ Connection error for '{title}': {type(e).__name__}")
                    emb = None
                except Exception as e:
                    api_errors.append({
                        "title": title,
                        "error": f"{type(e).__name__}: {str(e)[:100]}"
                    })
                    print(f"⚠️ API error for '{title}': {type(e).__name__}")
                    emb = None
            else:
                emb = existing_index.get(title, {}).get("embedding")

            # versioned meta
            updated_item = {
                **item,
                "embedding": emb,
                "_meta": {
                    "embedding_model": MODEL,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "flatten_version": FLATTEN_VERSION
                }
            }

            updated_data.append(updated_item)

        # remove old entries not in flatten
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved → {output_path} ({len(updated_data)} items)")

    # Report errors
    print("\n" + "="*70)
    if connection_errors:
        print(f"🔴 CONNECTION ERRORS ({len(connection_errors)} items):")
        for err in connection_errors[:10]:
            print(f"  - {err['title']}: {err['error']}")
        if len(connection_errors) > 10:
            print(f"  ... and {len(connection_errors) - 10} more")
    else:
        print("✅ No connection errors detected")
    
    if api_errors:
        print(f"\n⚠️ API ERRORS ({len(api_errors)} items):")
        for err in api_errors[:10]:
            print(f"  - {err['title']}: {err['error']}")
        if len(api_errors) > 10:
            print(f"  ... and {len(api_errors) - 10} more")
    else:
        print("✅ No API errors detected")
    print("="*70)

    print("\n🎉 All files processed.")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    process_all_files()
