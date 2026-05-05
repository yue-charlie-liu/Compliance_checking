import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata

from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz
from joblib import Parallel, delayed
from tqdm import tqdm

def load_text(path: str) -> str:
    """Load prompt text from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_mineru(path: str, words_per_chunk: int = 50_000, ocr: bool = False) -> list[str]:
    """Extract text from PDF using MinerU, grouping several pages per chunk."""
    input_filename = os.path.splitext(os.path.basename(path))[0]
    method = "ocr" if ocr else "auto"
    content_file = f'temp_mineru/{input_filename}/{method}/{input_filename}_content_list.json'
    md_file = f'temp_mineru/{input_filename}/{method}/{input_filename}.md'

    if not os.path.exists(content_file):
        os.makedirs('temp_mineru', exist_ok=True)
        cmd = ["mineru", "--path", path, "--output", "temp_mineru", "--method", method, "-f", "False"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Try fallback to python -m mineru if CLI is not available
            if importlib.util.find_spec("mineru") is not None:
                cmd = [sys.executable, "-m", "mineru", "--path", path, "--output", "temp_mineru", "--method", method, "-f", "False"]
                result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU extraction failed with return code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    if not os.path.exists(content_file):
        raise FileNotFoundError(
            f"Expected MinerU output file not found: {content_file}. "
            f"Make sure MinerU successfully produced the content list and that the method '{method}' is supported."
        )

    texts = []
    extracted_json = json.loads(load_text(content_file))

    previous_page_idx = 0
    current_chunk = f'<<<Page 1>>>\n'
    for item in extracted_json:
        if item['page_idx'] > previous_page_idx:
            previous_page_idx = item['page_idx']
            current_chunk += f"\n<<<PAGE {previous_page_idx+1}>>>\n"

        if item['type'] != 'text':
            continue
        current_chunk += f"\n{item['text']}\n"

        if len(current_chunk.split()) >= words_per_chunk:
            texts.append(current_chunk)
            current_chunk = f'<<<Page {previous_page_idx+1}>>>\n'

    texts.append(current_chunk)

    # Copy images
    os.makedirs(f'images/{input_filename}', exist_ok=True)
    os.system(f'cp -r "temp_mineru/{input_filename}/{method}/images/." "images/{input_filename}"')

    # Copy markdown
    os.makedirs(f'markdown', exist_ok=True)
    os.system(f'cp "{md_file}" "markdown/{input_filename}.md"')

    # os.system('rm -r temp_mineru')
    return texts

def chunk_prompt() -> str:
    return load_text('prompts/chunk.txt')

def user_message_for_chunk(chunk_text: str) -> str:
    return f'\nINPUT TEXT CHUNK BELOW\n-----\n{chunk_text}'

def merge_chunks(json_list: list[dict]) -> dict:
    processed_data = []
    for chunk in json_list:
        content = chunk.get('content', [])
        for item in content:
            if len(item['title']) > 200:
                print(f"Warning: Title too long ({len(item['title'])} chars): {item['title'][:100]}...")
            if processed_data and (item['title'] == '' or item['title'] == processed_data[-1]['title']):
                processed_data[-1]['text'] += '\n'+item['text']
            elif item['title'] != '':
                processed_data.append(item)
    return processed_data

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = text.replace("0", "o")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("，", ",")
    text = text.replace("；", ";")
    text = text.replace("•", " ")
    text = text.replace("\\mathrm", " ")
    text = text.replace("\\", " ")
    text = text.replace("$", " ")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-z]+", "", text)
    return text


def _iter_candidate_windows(content_list: List[Dict], idx: int) -> List[Tuple[int, str]]:
    """
    Build candidate windows for fuzzy heading matching.
    Returns tuples of (anchor_index, candidate_text).
    """
    windows: List[Tuple[int, str]] = []

    text0 = content_list[idx].get('text', '')
    windows.append((idx, text0))

    if idx + 1 < len(content_list):
        text1 = content_list[idx + 1].get('text', '')
        windows.append((idx + 1, text1))
        windows.append((idx, f"{text0} {text1}"))

    if idx + 2 < len(content_list):
        text2 = content_list[idx + 2].get('text', '')
        windows.append((idx + 2, text2))
        windows.append((idx, f"{text0} {content_list[idx + 1].get('text', '')} {text2}"))

    return windows


def _best_match_at_index(content_list: List[Dict], idx: int, title_norm: str) -> Tuple[float, int]:
    """
    Return the best fuzzy score and its anchor index for a given search index.
    """
    best_score = 0.0
    best_anchor_idx = idx
    title_len = max(len(title_norm), 1)

    for anchor_idx, candidate in _iter_candidate_windows(content_list, idx):
        candidate_norm = normalize(candidate)
        if not candidate_norm:
            continue

        # Avoid over-matching against very long merged blocks.
        if len(candidate_norm) > max(180, title_len * 9):
            continue

        partial = fuzz.partial_ratio(title_norm, candidate_norm)
        ratio = fuzz.ratio(title_norm, candidate_norm)
        score = 0.65 * partial + 0.35 * ratio

        # Strong boost for contained exact-normalized matches in reasonably-sized blocks.
        if title_norm in candidate_norm and len(candidate_norm) <= max(280, title_len * 6):
            score = max(score, 99.0)

        if score > best_score:
            best_score = score
            best_anchor_idx = anchor_idx

    return best_score, best_anchor_idx


def _neighbor_bounds(positions: List[Optional[int]], i: int, size: int) -> Tuple[int, int]:
    """Return search bounds [left, right) from nearest already-assigned neighbors."""
    left = 0
    for j in range(i - 1, -1, -1):
        if positions[j] is not None:
            left = positions[j] + 1
            break

    right = size
    for j in range(i + 1, len(positions)):
        if positions[j] is not None:
            right = positions[j]
            break

    return left, right


def _recover_missing_positions(
    content_list: List[Dict],
    data: List[Dict],
    heading_positions: List[Optional[int]],
    threshold: float,
    require_unused: bool = False,
    used_indices: Optional[set[int]] = None,
) -> None:
    """Fill missing heading indices using bounded local search between known neighbors."""
    for i, pos in enumerate(heading_positions):
        if pos is not None:
            continue

        title_norm = normalize(data[i]['title'])
        if not title_norm:
            continue

        left, right = _neighbor_bounds(heading_positions, i, len(content_list))
        if left >= right:
            continue

        best_score = 0.0
        best_idx = None

        for idx in range(left, right):
            score, anchor_idx = _best_match_at_index(content_list, idx, title_norm)
            if require_unused and used_indices is not None and anchor_idx in used_indices:
                continue
            if score > best_score:
                best_score = score
                best_idx = anchor_idx

        if best_idx is not None and best_score >= threshold:
            heading_positions[i] = best_idx
            if used_indices is not None:
                used_indices.add(best_idx)


def _enforce_monotonic_positions(heading_positions: List[Optional[int]]) -> None:
    """Ensure heading indices are non-decreasing; invalidate backward jumps."""
    last = -1
    for i, pos in enumerate(heading_positions):
        if pos is None:
            continue
        if pos < last:
            heading_positions[i] = None
            continue
        last = pos

def obtain_positions(filename: str, data: List[Dict], ocr: bool = False, similarity_threshold: float = 0.85, skip_toc: int = 15) -> List[Dict]:
    similarity_threshold *= 100
    input_filename = os.path.splitext(os.path.basename(filename))[0]
    method = "ocr" if ocr else "auto"
    output_filename = f'temp_mineru/{input_filename}/{method}/{input_filename}_content_list.json'
    if not os.path.exists(output_filename):
        os.system(f'mineru --path "{filename}" --output temp_mineru --method {method} -f False')
    with open(output_filename, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
        content_list = [
            item for item in content_list
            if item['type'] == 'text' or item['type'] == 'discarded'
        ]

    page_index = {}

    for i, item in enumerate(content_list):
        page = item['page_idx'] + 1
        page_index.setdefault(page, []).append((i, item))

    pages_sorted = sorted(page_index.keys())

    # -----------------------------
    # Detect dense pages (TOC)
    # -----------------------------

    page_heading_counts = {p: 0 for p in pages_sorted}

    for p in pages_sorted:
        items = page_index[p]

        for heading in data[:min(len(data), 30)]:  # sample early headings
            title_norm = normalize(heading["title"])
            if not title_norm:
                continue

            matched_on_page = False

            for i, _ in items:
                for _, candidate in _iter_candidate_windows(content_list, i):
                    candidate_norm = normalize(candidate)
                    if not candidate_norm:
                        continue
                    score = fuzz.partial_ratio(title_norm, candidate_norm)
                    if score >= similarity_threshold:
                        matched_on_page = True
                        break

                if matched_on_page:
                    break

            if matched_on_page:
                page_heading_counts[p] += 1

    dense_pages = {p for p, c in page_heading_counts.items() if c >= skip_toc}

    # -----------------------------
    # Sequential heading search
    # -----------------------------

    current_index = 0
    lookahead_window = 900
    heading_positions = []

    for heading in tqdm(data, desc="Title"):

        title = heading['title']
        found_index = None
        title_norm = normalize(title)
        if not title_norm:
            heading_positions.append(None)
            continue

        best_score_in_window = 0.0
        best_idx_in_window = None

        scan_end = min(len(content_list), current_index + lookahead_window)

        for idx in range(current_index, scan_end):

            item = content_list[idx]
            page = item['page_idx'] + 1

            best_score, best_anchor_idx = _best_match_at_index(content_list, idx, title_norm)

            page_threshold = similarity_threshold
            if page in dense_pages:
                page_threshold = max(similarity_threshold, 98)

            if best_score > best_score_in_window:
                best_score_in_window = best_score
                best_idx_in_window = best_anchor_idx

            # Strong early accept within nearby context.
            if best_score >= page_threshold + 3:
                found_index = best_anchor_idx
                break

        # Prefer the strongest nearby match if it is good enough.
        if found_index is None and best_idx_in_window is not None and best_score_in_window >= similarity_threshold:
            found_index = best_idx_in_window

        # Fallback to full forward scan if nearby context failed.
        if found_index is None:
            for idx in range(scan_end, len(content_list)):

                item = content_list[idx]
                page = item['page_idx'] + 1

                best_score, best_anchor_idx = _best_match_at_index(content_list, idx, title_norm)

                page_threshold = similarity_threshold
                if page in dense_pages:
                    page_threshold = max(similarity_threshold, 98)

                if best_score >= page_threshold:

                    found_index = best_anchor_idx
                    break

        if found_index is None:
            heading_positions.append(None)
            continue

        heading_positions.append(found_index)
        current_index = found_index

    # -----------------------------
    # Fallback recovery for misses
    # -----------------------------

    _recover_missing_positions(
        content_list=content_list,
        data=data,
        heading_positions=heading_positions,
        threshold=similarity_threshold
    )

    # Safety pass: invalidate backward jumps and recover once more locally.
    _enforce_monotonic_positions(heading_positions)
    _recover_missing_positions(
        content_list=content_list,
        data=data,
        heading_positions=heading_positions,
        threshold=similarity_threshold,
    )
    _enforce_monotonic_positions(heading_positions)

    # -----------------------------
    # Build element ranges
    # -----------------------------

    no_matches = []

    for i, heading in enumerate(data):

        start = heading_positions[i]

        if start is None:
            data[i]['position'] = []
            no_matches.append(heading["title"])
            continue

        end = len(content_list)

        for j in range(i+1, len(heading_positions)):
            if heading_positions[j] is not None:
                end = heading_positions[j]
                break

        position = []

        first = content_list[start]

        x0, y0, x1, y1 = first['bbox']

        position.append({
            "type": "title",
            "bbox": [x0/10, y0/10, x1/10, y1/10],
            "page": first['page_idx'] + 1
        })

        for k in range(start+1, end):

            item = content_list[k]

            if item['type'] == 'discarded':
                continue

            x0, y0, x1, y1 = item['bbox']

            position.append({
                "type": "text",
                "bbox": [x0/10, y0/10, x1/10, y1/10],
                "page": item['page_idx'] + 1
            })

        data[i]["page"] = first["page_idx"] + 1
        data[i]["position"] = position

    if no_matches:
        log_file = f'temp_mineru/{input_filename}/{method}/warnings.log'
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(no_matches))
            print(f'{len(no_matches)} warnings logged to {log_file}')
    return data


class CompactPositionEncoder(json.JSONEncoder):
    """Keep `position` arrays compact (single-line) while preserving global indent."""

    _POS_TOKEN_PREFIX = "__POS_TOKEN_"

    def _process(self, obj, replacements):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if key == 'position':
                    token = f"{self._POS_TOKEN_PREFIX}{len(replacements)}__"
                    replacements[token] = json.dumps(
                        value,
                        ensure_ascii=self.ensure_ascii,
                        separators=(',', ':'),
                    )
                    out[key] = token
                else:
                    out[key] = self._process(value, replacements)
            return out

        if isinstance(obj, list):
            return [self._process(item, replacements) for item in obj]

        return obj

    def _encode_compact(self, obj, _one_shot=False):
        replacements = {}
        processed = self._process(obj, replacements)
        raw = ''.join(json.JSONEncoder.iterencode(self, processed, _one_shot))

        if not replacements:
            return raw

        return re.sub(
            r'"(__POS_TOKEN_\d+__)"',
            lambda m: replacements.get(m.group(1), m.group(0)),
            raw,
        )

    def encode(self, obj):
        return self._encode_compact(obj, _one_shot=True)

    def iterencode(self, obj, _one_shot=False):
        yield self._encode_compact(obj, _one_shot=_one_shot)
