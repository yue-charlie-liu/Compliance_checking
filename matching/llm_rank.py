import json
import os
import re
import time
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI

# -----------------------
# Config
# -----------------------
BASE_DIR = Path("./top_matches")
OUTPUT_DIR = Path("./reranked")
MODEL ='gpt-5.2'

MAX_WORKERS = 3
MAX_RETRIES = 3
SLEEP = 2

# -----------------------
# API
# -----------------------
sys.path.append(os.path.abspath("../../"))
from utils.api_keys import get_openai_api_key

client = OpenAI(api_key=get_openai_api_key())

# -----------------------
# Prompt
# -----------------------
def build_prompt(p1, p2):
    return f"""
        You are given one regulatory provision (Main) and exactly 10 related provisions.
        Your task is to rank the 10 related provisions by their similarity to the Main provision.

        [Instructions]
        - Rank based only on the meaning of the texts.
        - Do NOT use external knowledge.
        - Do NOT invent new items.
        - Every related provision must receive exactly one unique rank from 1 to 10.
        - Rank 1 = most similar, Rank 10 = least similar.
        - No ties are allowed.
        - Use the titles exactly as provided.

        [Output Format]
        Return ONLY a JSON array of 10 objects in the following format:
        [
            {{
            "title": "...",
            "rank": 1
            }},
            ...
        ]

        [Main provision]
        {p1}

        [Related provisions]
        {p2}
        """

# -----------------------
# Helpers
# -----------------------
def safe_parse_rank_json(text):
    """Parse JSON array from LLM output, removing code blocks"""
    try:
        text = re.sub(r"```.*?```", "", text, flags=re.S).strip()
        match = re.search(r"\[[\s\S]*?\]", text)
        if not match:
            return None, "No JSON array found"
        return json.loads(match.group(0)), None
    except Exception as e:
        return None, str(e)

CACHE = {}

def call_llm(source, targets):
    """Call LLM and return parsed ranking"""
    key = hashlib.md5(
        (source + json.dumps(targets, ensure_ascii=False)).encode()
    ).hexdigest()

    if key in CACHE:
        return CACHE[key], None

    prompt = build_prompt(source, targets)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                reasoning={"effort": "low"}
            )
            parsed, error = safe_parse_rank_json(resp.output_text)
            if parsed:
                CACHE[key] = parsed
                return parsed, None
            else:
                print(f"⚠️ Parse error: {error}, retry {attempt}/{MAX_RETRIES}")
                time.sleep(SLEEP * attempt)
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}, retry {attempt}/{MAX_RETRIES}")
            time.sleep(SLEEP * attempt)

    return None, "Max retries exceeded"

def strip_text_fields(data):
    """Remove all 'text' fields before saving"""
    for src in data:
        src.pop("text", None)
        top_matches = src.get("top_matches", {})
        for group_name, matches in top_matches.items():
            for m in matches:
                m.pop("text", None)

# -----------------------
# Data loading
# -----------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def build_title_to_text_map(folder):
    """Build mapping title -> text"""
    mapping = {}
    for file in folder.glob("*.json"):
        data = load_json(file)
        for item in data:
            title = item.get("title")
            text = item.get("text")
            if title and text:
                mapping[title] = text
    return mapping

# -----------------------
# Task builder
# -----------------------
def build_tasks_from_file(file_path, title_text_map):
    data = load_json(file_path)
    tasks = []

    for src in data:
        source_title = src["title"]
        source_text = src["text"]

        top_matches = src.get("top_matches", {})

        for group_name, matches in top_matches.items():
            targets = []
            for m in matches:
                t_title = m["title"]
                if t_title not in title_text_map:
                    print(f"⚠️ Missing title: {t_title}")
                    continue
                targets.append({"title": t_title, "text": title_text_map[t_title]})

            task = {
                "source_title": source_title,
                "source_text": source_text,
                "targets": targets,
                "group_name": group_name,
                "matches_ref": matches
            }
            tasks.append(task)

    return data, tasks

# -----------------------
# Worker with retry
# -----------------------
def process_task_with_retry(task, max_retries=3):
    """Process task, retry if missing ranks"""
    for attempt in range(1, max_retries + 1):
        source_block = f"{task['source_title']}\n\n{task['source_text']}"
        targets = task["targets"]

        result, error = call_llm(source_block, targets)
        matches = task["matches_ref"]

        if not result:
            for m in matches:
                m["llm_rank_error"] = error
            print(f"⚠️ Task failed (attempt {attempt}/{max_retries}): {error}")
            time.sleep(SLEEP * attempt)
            continue

        # check if there is missing rank
        rank_map = {r["title"]: r.get("rank") for r in result if "rank" in r}
        missing_titles = [m["title"] for m in matches if m["title"] not in rank_map]

        if missing_titles:
            for m in matches:
                if m["title"] in missing_titles:
                    m["llm_rank_error"] = "Missing from LLM output"
            print(f"⚠️ Missing ranks for: {missing_titles} (attempt {attempt}/{max_retries})")
            time.sleep(SLEEP * attempt)
            continue

        for m in matches:
            title = m["title"]
            m["llm_rank"] = rank_map[title]
            m.pop("llm_rank_error", None)

        return task, True

    # still failed exceeding max retry
    for m in matches:
        m["llm_rank_error"] = "Missing from LLM output"
    return task, False

# -----------------------
# Main
# -----------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_files = list((BASE_DIR / "Road").glob("*.json")) + list((BASE_DIR / "WHS").glob("*.json"))
    
    # Skip specific WHS files
    skip_files = {"NSW - Work Health and Safety Regulation 2017.json", "Victoria - Occupational Health and Safety Regulations 2017.json"}
    all_files = [f for f in all_files if f.name not in skip_files]
    
    print(f"Found {len(all_files)} files")

    for file in all_files:
        print(f"\n📄 Processing {file.name}")
        folder = file.parent
        title_text_map = build_title_to_text_map(folder)
        original_data, tasks = build_tasks_from_file(file, title_text_map)
        print(f"→ Built {len(tasks)} rerank tasks")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_task_with_retry, t, MAX_RETRIES) for t in tasks]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing tasks ({file.name})"):
                task, success = future.result()
                if not success:
                    # still failed exceeding max retry
                    for m in task["matches_ref"]:
                        m["llm_rank_error"] = "Max retries exceeded or missing ranks"

        # don't need to store text
        strip_text_fields(original_data)

        out_path = OUTPUT_DIR / file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved to {out_path}")

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()