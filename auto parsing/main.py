"""
Uses MinerU to extract text from a PDF in chunks, then calls OpenAI's API
for converting the extracted text into structured JSON.
"""
import argparse
import json
import os
import time
from google import genai
from typing import Any, List, Dict, Tuple
from openai import OpenAI
from joblib import Parallel, delayed

from config import OPENAI_API_KEY, GOOGLE_API_KEY, MODEL, TEMPERATURE, MAX_RETRIES, WORDS_PER_CHUNK, MODEL_LOW
from schema import FlatSchema, TOCSchema, MetadataSchema
from helper import extract_mineru, chunk_prompt, user_message_for_chunk, merge_chunks, obtain_positions, load_text, CompactPositionEncoder


# client = OpenAI(api_key=OPENAI_API_KEY)

def call_gemini_api(
    system_prompt: str,
    user_prompt: str | None = None,
    schema = None,
    model: str = 'gemini-2.5-flash-lite',
    retries: int = MAX_RETRIES,
) -> Dict[str, Any] | None:
    """
    Calls Gemini using the new models.generate_content API.
    Enforces JSON output via response_json_schema (Pydantic).
    Returns parsed dict or None on failure.
    """

    for attempt in range(retries + 1):
        try:
            contents = []

            # Gemini does not have explicit roles; system prompt goes first
            if system_prompt:
                contents.append(system_prompt)

            if user_prompt:
                contents.append(user_prompt)

            _client = genai.Client(api_key=GOOGLE_API_KEY)
            response = _client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "response_mime_type": "application/json" if schema else "text/plain",
                    "response_json_schema": schema.model_json_schema() if schema else None,
                },
            )

            if schema:
                parsed = json.loads(response.text)
                return schema.model_validate(parsed).model_dump()

            return response.text

        except Exception as e:
            print(f"[call_gemini_api] attempt {attempt} failed: {e}")
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)

def call_openai_api(system_prompt: str, user_prompt: str = None, schema = None, model: str = MODEL, effort: str = "low", retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """
    Calls the model, requesting the function 'extract_legal_segments' to be returned.
    Returns the parsed JSON (dict) from the model function call.
    """
    for attempt in range(retries + 1):
        try:
            messages = [{"role": "system", "content": system_prompt},]
            if user_prompt is not None:
                messages.append({"role": "user", "content": user_prompt})
            _client = OpenAI(api_key=OPENAI_API_KEY)
            response = _client.responses.parse(
                model=model,
                input=messages,
                temperature=TEMPERATURE,
                max_output_tokens=128_000,      # adjust as needed
                reasoning={"effort": effort}, # can be 'minimal', 'low', 'medium', 'high'
                text_format = schema,
                # service_tier="flex" # 50% less cost
            )
            if response.incomplete_details is not None:
                print(f"[call_openai_api] Warning: Reason for incomplete response: {response.incomplete_details}")
            return response.output_parsed.model_dump()

        except Exception as e:
            print(f"[call_openai_extract] attempt {attempt} failed: {e}")
            if attempt == retries: return None
            time.sleep(2 ** attempt)

def process_chunk(i, chunk, debug=True, debug_dir=''):
    try:
        with open(f'{debug_dir}/processed_{i}.json', 'r', encoding='utf-8') as pf:
            return json.load(pf)
    except FileNotFoundError:
        pass  # File doesn't exist, proceed to process

    if debug:
        with open(f'{debug_dir}/ocr_{i}.txt', 'w', encoding='utf-8') as cf:
            cf.write(chunk)

    result = call_openai_api(
        system_prompt=chunk_prompt(),
        user_prompt=user_message_for_chunk(chunk),
        schema=FlatSchema,
        model=MODEL,
        effort="medium"
    )
    if result is None:
        print(f"Chunk {i} failed. Failing silently.")
    elif debug:
        with open(f'{debug_dir}/processed_{i}.json', 'w', encoding='utf-8') as pf:
            json.dump(result, pf, indent=4, ensure_ascii=True)
    return result

def get_nested_json(json_list: List[Dict[str, Any]], retries: int = MAX_RETRIES) -> Tuple[List[Dict[str, Any]], List[str]]:
    prompt = load_text('prompts/table_of_contents.txt').replace('{NUM_SEGMENTS}', str(len(json_list)))
    toc = call_openai_api(
        system_prompt=prompt,
        user_prompt=str([{'index': i+1, 'title': x['title'][:100]} for i, x in enumerate(json_list)]),
        schema=TOCSchema,
        model='gpt-5.2',
        effort="medium"
    )
    if toc is None:
        raise RuntimeError(
            "Failed to obtain Table of Contents from OpenAI after retries. "
            "Check network connectivity, API credentials, and model availability."
        )
    if len(toc['content']) != len(json_list):
        if retries > 0:
            print(f"TOC length {len(toc['content'])} does not match number of segments {len(json_list)}; retrying...{retries} attempts left")
            return get_nested_json(json_list, retries=retries-1)
        else:
            raise ValueError(f"TOC length {len(toc['content'])} does not match number of segments {len(json_list)}")

    if 'missing' in toc and toc['missing']:
        print(f"Warning: Expected sections not found\n\n{toc['missing']}")

    types = set([entry['type'] for entry in toc['content'] if entry['action'] == 'add'])
    nested, parents = [], []
    for item, entry in zip(json_list, toc['content']):
        if entry['action'] == 'remove':
            continue
        if entry['action'] == 'merge' and entry['level'] > 1 and parents:
            parents[entry['level'] - 2]['text'] += '\n' + item['text']
            continue
        node = {**item, 'id': entry['unique_id'], 'type': entry['type'], 'subsegments': []}
        parents = parents[:entry['level'] - 1]
        if entry['level'] == 1:
            nested.append(node)
        else:
            parents[-1]['subsegments'].append(node)
        parents.append(node)
    return nested, list(types)


def get_metadata(metadata_text: str) -> Dict[str, Any]:
    metadata_prompt = load_text('prompts/metadata.txt')
    metadata_input = metadata_prompt + metadata_text
    return call_openai_api(metadata_input, schema=MetadataSchema, model=MODEL_LOW, effort="minimal")

def run_pipeline(pdf_path: str, out_path: str, ocr: bool = False, debug: bool = True):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = f'{out_dir}/chunks/{os.path.splitext(os.path.basename(out_path))[0]}'
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    chunks = extract_mineru(pdf_path, words_per_chunk=WORDS_PER_CHUNK, ocr=ocr)
    results = Parallel(n_jobs=os.cpu_count(), verbose=10)(
        delayed(process_chunk)(i, chunk, debug=debug, debug_dir=debug_dir)
        for i, chunk in enumerate(chunks, start=1)
    )
    json_list = [res for res in results if res is not None]

    # Save output JSON
    flat_data = merge_chunks(json_list)
    print("Obtaining page positions...")
    flat_data = obtain_positions(pdf_path, flat_data, ocr=ocr)
    print("Obtaining Table of Contents and nesting structure...")
    nested_data, types = get_nested_json(flat_data)
    metadata_text = ''.join(chunks[:min(6, len(chunks))])  # first 6 chunks for metadata
    print("Obtaining Metadata...")
    processed_data = get_metadata(metadata_text=metadata_text)
    if processed_data is None:
        raise RuntimeError(
            "Failed to obtain metadata from OpenAI after retries. "
            "Check network connectivity, API credentials, and model availability."
        )
    processed_data['numberedSegments'] = types
    processed_data['content'] = nested_data
    print(f"Writing output to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        json.dump(processed_data, out_file, indent=4, ensure_ascii=True, cls=CompactPositionEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--out", help="Path to the output markdown file (defaults to PDF name)")
    parser.add_argument("--ocr", type=bool, help="Force OCR before parsing")
    args = parser.parse_args()
    output_filename = args.out or os.path.splitext(args.pdf)[0] + ".json"
    ocr = True if args.ocr is not None else False
    print("OCR mode:", ocr)

    run_pipeline(args.pdf, output_filename, ocr=ocr)
