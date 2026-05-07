import os
import json
import numpy as np
from tqdm import tqdm
import hashlib

# -----------------------
# Hash function
# -----------------------
def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()


# -----------------------
# Load embeddings from a folder
# -----------------------
def load_embeddings(dirpath: str):
    """
    Load all JSON files with embeddings in the folder.
    Returns dict: {state_name: [provision dicts with 'embedding']}
    """
    data = {}
    for filename in os.listdir(dirpath):
        if not filename.endswith(".json"):
            continue
        state = filename.replace(".json", "")
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            provisions = json.load(f)
            # Add hash_id to each provision for identification
            for prov in provisions:
                prov['hash_id'] = f"{state.lower()}_{hash_id(prov.get('text',''))}"
            data[state] = provisions
    return data


# -----------------------
# Cosine similarity
# -----------------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# -----------------------
# Compact JSON dump
# -----------------------
def compact_dump(obj, output_path: str, indent=2):
    # """
    # Dump JSON to file, but collapse arrays into single lines to save space.
    # """
    # import re
    # text = json.dumps(obj, indent=indent, ensure_ascii=False)
    # # Collapse arrays: [1, 2, 3] instead of multi-line
    # text = re.sub(r"\[\s*((?:.|\n)*?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(text)
    """
    Safely dump JSON to file:
    - Does NOT modify any string content
    - Unicode-safe
    - Keeps arrays/text/title exactly as-is
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# -----------------------
# Compute matches from one list to another
# -----------------------
def compute_matches_from_to(source_provisions: list, target_provisions: list, source_name: str, target_name: str, top_k: int = 10, output_dir="./top_matches"):
    """
    Compute top_k matches for each provision in source_provisions against target_provisions.
    """
    # Filter valid provisions with embeddings
    source_valid = [p for p in source_provisions if p.get('embedding') is not None]
    target_valid = [p for p in target_provisions if p.get('embedding') is not None]
    
    if not source_valid:
        print(f"⚠️ No valid embeddings in source {source_name}, skipping")
        return
    if not target_valid:
        print(f"⚠️ No valid embeddings in target {target_name}, skipping")
        return
    
    # Check embedding lengths
    source_lengths = set(len(p['embedding']) for p in source_valid)
    target_lengths = set(len(p['embedding']) for p in target_valid)
    
    if len(source_lengths) > 1:
        print(f"⚠️ Warning: source embeddings of different lengths: {source_lengths}")
        common_len = max(source_lengths, key=lambda x: sum(len(p['embedding'])==x for p in source_valid))
        source_valid = [p for p in source_valid if len(p['embedding'])==common_len]
    
    if len(target_lengths) > 1:
        print(f"⚠️ Warning: target embeddings of different lengths: {target_lengths}")
        common_len = max(target_lengths, key=lambda x: sum(len(p['embedding'])==x for p in target_valid))
        target_valid = [p for p in target_valid if len(p['embedding'])==common_len]
    
    # Create normalized matrices
    source_matrix = np.array([p['embedding'] for p in source_valid])
    source_matrix = source_matrix / np.linalg.norm(source_matrix, axis=1, keepdims=True)
    
    target_matrix = np.array([p['embedding'] for p in target_valid])
    target_matrix = target_matrix / np.linalg.norm(target_matrix, axis=1, keepdims=True)
    
    # Compute matches
    result = []
    for i, item in enumerate(tqdm(source_valid, desc=f"{source_name} → {target_name}")):
        vec = source_matrix[i]
        sims = target_matrix @ vec
        
        matches = [
            {"hash_id": p['hash_id'], "title": p.get('title',''), "score": float(score)}
            for p, score in zip(target_valid, sims)
        ]
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove embedding before saving
        out = {k: v for k, v in item.items() if k != "embedding"}
        out["top_matches"] = {target_name: matches[:top_k]}
        result.append(out)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{source_name}.json")
    compact_dump(result, output_file)
    print(f"Saved → {output_file} ({len(result)} items)")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Load the two specific files
    with open("./embedding/csiro_Electrical.json", "r", encoding="utf-8") as f:
        csiro_data = json.load(f)
        # Add hash_id to each provision
        for prov in csiro_data:
            prov['hash_id'] = f"csiro_{hash_id(prov.get('text',''))}"
    
    with open("./embedding/model WHS regulations.json", "r", encoding="utf-8") as f:
        whs_data = json.load(f)
        # Add hash_id to each provision
        for prov in whs_data:
            prov['hash_id'] = f"whs_{hash_id(prov.get('text',''))}"
    
    # Compute matches from csiro to whs
    compute_matches_from_to(csiro_data, whs_data, "csiro_Electrical", "model_WHS_regulations", top_k=10, output_dir="./top_matches")
