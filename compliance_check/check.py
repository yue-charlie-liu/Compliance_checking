import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from joblib import Parallel, delayed

# Ensure repo root is importable for utils.api_keys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.api_keys import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

TOP_MATCH_FILE = ROOT_DIR / "matching" / "top_matches" / "csiro_Electrical.json"
FLATTENED_DIR = ROOT_DIR / "chunking" / "flattened"
OUTPUT_DIR = Path(__file__).resolve().parent / "result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY = 3
MODEL = "gpt-5.2"
N_JOBS = 1  # Use all available CPU cores


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_flatten_file(list_name: str, available_files: List[Path]) -> Optional[Path]:
    normalized = normalize_name(list_name)
    exact = [p for p in available_files if normalize_name(p.stem) == normalized]
    if exact:
        return exact[0]

    # Try replacing underscores with spaces and vice versa
    normalized_alt = normalize_name(list_name.replace("_", " "))
    alt = [p for p in available_files if normalize_name(p.stem) == normalized_alt]
    if alt:
        return alt[0]

    return None


def build_flattened_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    files = list(FLATTENED_DIR.glob("*.json"))
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for file in files:
        data = load_json(file)
        title_map: Dict[str, Dict[str, Any]] = {}
        for item in data:
            title = item.get("title", "").strip()
            if title:
                title_map[title] = item
                title_map[normalize_name(title)] = item
        index[file.name] = title_map
    return index

def call_compliance_gpt(source_text: str, target_text: str, client: OpenAI) -> Dict[str, Any]:
    """Check compliance between source and target texts using GPT-5.2."""
    prompt = f"""You are a compliance analyst. 
Follow these two steps to analyze the relationship between the POLICY and PROVISION texts.

POLICY TEXT:
{source_text}

PROVISION TEXT:
{target_text}

STEP 1: Determine if these two texts are functionally related such that a compliance check is necessary. 
(e.g., Is the source trying to fulfill an obligation defined in the target?)

STEP 2: If related, perform the compliance check.

Return only JSON with these fields:
- is_relevant: true or false (Result of STEP 1)
- compliant: true, false, or null (null if is_relevant is false)
- reasoning: explanation of compliance status (null if is_relevant is false)
- remediation: suggested action if not compliant; otherwise empty
"""
    
    # time.sleep(random.uniform(0, 2))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_output_tokens=1000,
            )
            text = response.output_text.strip()
            
            # Parse JSON response
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        result = json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")
                else:
                    raise ValueError(f"No JSON found in response: {text[:200]}...")
            
            # Validate required fields
            required_fields = ["is_relevant", "compliant", "reasoning", "remediation"]
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")
            
            # Handle non-relevant cases
            if not result.get("is_relevant", False):
                return {
                    "is_relevant": False,
                    "compliant": None,
                    "reasoning": result.get("reasoning", "No regulatory relevance identified."),
                    "remediation": ""
                }
            
            return result

        except Exception as exc:
            error_msg = f"GPT check failed (attempt {attempt}/{MAX_RETRIES}): {exc}"
            print(f"⚠️ {error_msg}")
            if attempt == MAX_RETRIES:
                # Return error result instead of raising
                return {
                    "is_relevant": None,
                    "compliant": None,
                    "reasoning": f"Error after {MAX_RETRIES} attempts: {str(exc)}",
                    "remediation": ""
                }
            time.sleep(RETRY_DELAY)

def process_single_check(task: Tuple[str, str, str, str, float], api_key: str) -> Dict[str, Any]:
    """Process a single compliance check task in parallel."""
    source_text, target_text, target_file, target_title, match_score = task
    
    # Create client in worker process to avoid pickling issues
    worker_client = OpenAI(api_key=api_key)
    
    compliance = call_compliance_gpt(source_text, target_text, worker_client)
    
    return {
        "target_file": target_file,
        "target_title": target_title,
        "target_text": target_text,
        "match_score": match_score,
        "compliance": compliance,
    }


def run_check() -> None:
    # Use the already loaded API key
    api_key = OPENAI_API_KEY
    
    top_matches = load_json(TOP_MATCH_FILE)
    flattened_index = build_flattened_index()
    available_flatten_files = list(FLATTENED_DIR.glob("*.json"))

    results: List[Dict[str, Any]] = []
    all_tasks: List[Tuple[Dict[str, Any], Tuple[str, str, str, str, float]]] = []

    # Collect all tasks
    for item in top_matches:
        source_title = item.get("title", "")
        source_text = item.get("text", "")
        if not source_text:
            continue

        top_matches_data = item.get("top_matches", {})
        if not isinstance(top_matches_data, dict):
            continue

        for list_name, matches in top_matches_data.items():
            target_file = find_flatten_file(list_name, available_flatten_files)
            if target_file is None:
                print(f"⚠️ Could not find flattened file for list name '{list_name}'")
                continue

            target_index = flattened_index.get(target_file.name, {})
            for match in matches:
                target_title = match.get("title", "").strip()
                target_item = target_index.get(target_title) or target_index.get(normalize_name(target_title))
                if target_item is None:
                    print(f"⚠️ Could not find title '{target_title}' in {target_file.name}")
                    continue

                target_text = target_item.get("text", "")
                if not target_text:
                    continue

                task = (source_text, target_text, target_file.name, target_title, match.get("score", 0.0))
                all_tasks.append((item, task))

    print(f"Collected {len(all_tasks)} compliance check tasks")

    # Process tasks in parallel
    if all_tasks:
        print(f"Processing in parallel with {N_JOBS} jobs...")
        task_results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(process_single_check)(task, api_key) for _, task in all_tasks
        )

        # Group results back by source item
        item_results_map: Dict[str, Dict[str, Any]] = {}
        for (item, _), result in zip(all_tasks, task_results):
            source_title = item.get("title", "")
            if source_title not in item_results_map:
                item_results_map[source_title] = {
                    "source_title": source_title,
                    "source_text": item.get("text", ""),
                    "matches": [],
                }
            item_results_map[source_title]["matches"].append(result)

        results = list(item_results_map.values())

    output_file = OUTPUT_DIR / "check_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    run_check()
