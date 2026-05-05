import json
import os
from typing import List
from main import call_openai_api

def extract_sections(filename: str, save=True, save_dir = 'test/5 verify headings/') -> List[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)['content']
    sections = []
    for item in data:
        sections.append(item['title'])
        if 'subsegments' in item and item['subsegments']:
            sections.extend(extract_sections(item['subsegments'], save=False))
    if save:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        save_path = os.path.join(save_dir, f'{base_name}_sections.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            for section in sections:
                f.write(section + '\n')
    return sections

def verify_headings(sections: List[str], save=True, save_dir='test/5 verify headings/', out_filename='verification_results.txt') -> List[str]:
    prompt = '''Please identify errors in section headings extracted from a document. Errors can occur in two ways:
    1) Number missing - If as per the numbering any heading is missing or duplicated
    2) Text missing - If the heading only contains only numbering and missed heading text

Return only a single JSON object as below

{"number missing": ["Part 8 Division 2", "304A", "304Q"], "text missing": ["120", "Part 11 Division 1"]}

Here are the section headings:
'''
    for section in sections:
        prompt += f"- {section}\n"
    response = call_openai_api(prompt, model='gpt-5.2', effort='high')
    if save:
        save_path = os.path.join(save_dir, out_filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response)
    return response

if __name__ == '__main__':
    vic_json_file = 'output/Australia road regulation/VIC - Road Safety Road Rules 2017.json'
    mic_json_file = 'output/human study/MCL-CHAP257.json'

    vic_sections = extract_sections(vic_json_file)
    mic_sections = extract_sections(mic_json_file)

    vic_verification = verify_headings(vic_sections, out_filename='VIC_verification_results.txt')
    mic_verification = verify_headings(mic_sections, out_filename='MIC_verification_results.txt')
