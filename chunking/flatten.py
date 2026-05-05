import json
from pathlib import Path

INPUT_DIR = Path("./output")
OUTPUT_DIR = Path("./flattened")

OUTPUT_DIR.mkdir(exist_ok=True)

# Only keep these types
VALID_TYPES = {"Article", "Subsection", "Rule", "Regulation"}

def extract_by_type(nodes, flattened):
    for node in nodes:
        node_type = node.get("type")

        # if current node type matches, collect it
        if node_type in VALID_TYPES:
            flattened.append({
                "id": node.get("id"),
                "title": node.get("title"),
                "text": node.get("text"),
                "type": node_type,
                "page": node.get("page"),
                "position": node.get("position"),
            })

        # continue recursion into subsegments regardless of current match
        for child in node.get("subsegments", []):
            extract_by_type([child], flattened)


for input_path in INPUT_DIR.glob("*.json"):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    extract_by_type(data.get("content", []), flattened)

    output_path = OUTPUT_DIR / f"{input_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"✅ {input_path.name}: {len(flattened)} extracted → {output_path}")

print("\n🎉 All files processed.")
