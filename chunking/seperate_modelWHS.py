import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

def is_valid_sequence(prev_num: str, curr_num: str) -> bool:
    """
    Verify if curr_num is a valid successor of prev_num.
    Supported sequences:
    1. Numeric increment: 689 -> 690
    2. Letter increment: 529C -> 529D
    3. Letter extension: 529C -> 529CA
    4. Initial number: prev_num is None
    """
    if prev_num is None:
        return True
    
    # Extract numeric and letter suffix parts (e.g., 689A -> 689, A)
    prev_m = re.match(r'(\d+)([A-Z]*)', prev_num)
    curr_m = re.match(r'(\d+)([A-Z]*)', curr_num)
    
    if not prev_m or not curr_m:
        return False
    
    p_int, p_suffix = int(prev_m.group(1)), prev_m.group(2)
    c_int, c_suffix = int(curr_m.group(1)), curr_m.group(2)
    
    # Case 1: Numeric increment (e.g., 689 -> 690)
    if c_int == p_int + 1 and not c_suffix:
        return True
    
    # Case 2: Same number, handle letter suffixes
    if c_int == p_int:
        # Extension (e.g., 529 -> 529A or 529C -> 529CA)
        if c_suffix.startswith(p_suffix) and len(c_suffix) > len(p_suffix):
            return True
        # Same length letter increment (e.g., 529C -> 529D)
        if len(p_suffix) == len(c_suffix) and p_suffix < c_suffix:
            return True
            
    # Case 3: Handle possible gaps in regulation numbering (e.g., 689 -> 700)
    # If number increases significantly with no suffix, usually valid
    if c_int > p_int and not c_suffix:
        return True

    return False

def split_regulation_text(full_text: str) -> List[tuple[str, str, str]]:
    """
    Parse text and extract Regulations that follow sequence logic.
    """
    # Match regulation number + title at line start
    # Number format: digits + optional letters (e.g., 689A), title must start with uppercase
    potential_matches = list(re.finditer(r'^(\d+[A-Z]*)\s+([A-Z].+?)$', full_text, re.MULTILINE))
    
    if not potential_matches:
        return []

    results = []
    valid_matches = []
    prev_num = None

    # First pass: Keep only matches with valid number sequences
    for m in potential_matches:
        curr_num = m.group(1)
        if is_valid_sequence(prev_num, curr_num):
            valid_matches.append(m)
            prev_num = curr_num

    # Second pass: Extract content based on validated positions
    for i in range(len(valid_matches)):
        m = valid_matches[i]
        reg_num = m.group(1)
        reg_title = f"{reg_num} {m.group(2).strip()}"
        
        start_pos = m.end()
        # All content before next Regulation is the current content
        end_pos = valid_matches[i+1].start() if i+1 < len(valid_matches) else len(full_text)
        
        reg_content = full_text[start_pos:end_pos].strip()
        results.append((reg_num, reg_title, reg_content))

    return results

def process_node(node: Dict[str, Any], parent_id: str = "") -> Dict[str, Any]:
    """
    Process single node (Part/Division/Subdivision), split embedded Regulations.
    """
    if not node.get('text'):
        return node
    
    text = node['text'].strip()
    regulations = split_regulation_text(text)
    
    if not regulations:
        return node
    
    print(f"  [Fixed] Found {len(regulations)} regulations in {node.get('type')} '{node.get('title')}'")
    
    new_subsegments = []
    for reg_num, reg_title, reg_content in regulations:
        reg_id = f"{node.get('id', parent_id)}_reg_{reg_num}"
        new_subsegments.append({
            "title": reg_title,
            "text": reg_content,
            "page": node.get('page', 0),
            "position": [],  # Coordinates become invalid after splitting
            "id": reg_id,
            "type": "Regulation",
            "subsegments": []
        })
    
    # Clear original node text, mount extracted content to subsegments
    node['text'] = ""
    # If original subsegments exist, merge them
    node['subsegments'] = new_subsegments + (node.get('subsegments', []))
    
    return node

def process_content_recursive(content: List[Dict[str, Any]], parent_id: str = "") -> List[Dict[str, Any]]:
    """
    Recursively traverse the entire JSON tree.
    """
    TARGET_TYPES = {'Part', 'Subdivision', 'Division'}
    
    for item in content:
        current_id = item.get('id', parent_id)
        
        # Depth-first traversal
        if item.get('subsegments'):
            item['subsegments'] = process_content_recursive(item['subsegments'], current_id)
        
        # Process current node
        if item.get('type') in TARGET_TYPES:
            process_node(item, current_id)
            
    return content

def process_json_file(input_path: str, output_path: str) -> None:
    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'content' in data:
        data['content'] = process_content_recursive(data['content'])
    
    print(f"Saving: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=True)
    print("Cleanup Complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python seperate_div_prov.py <input_json> [output_json]")
        sys.exit(1)
    
    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else in_file
    process_json_file(in_file, out_file)