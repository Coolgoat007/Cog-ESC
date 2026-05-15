#!/usr/bin/env python3
"""
Fix claude_sonnet (528 empty) and regenerate gpt4o_mini (brevity contaminated).
Supports resume: skips already-valid entries.
"""
import os, json, sys
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/root/baseline/CSO-main')
from generate_strong_baseline_responses import SUPPORTER_SYS_PROMPT, parse_llama_factory_prompt

client = OpenAI(api_key=os.getenv('openai_key'), base_url=os.getenv('openai_url'))
input_file = '/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl'

MODELS = {
    'claude_sonnet': 'claude-sonnet-4-6',
    'gpt4o_mini': 'gpt-4o-mini',
}

def call_api(model_id, chat_history):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'system', 'content': SUPPORTER_SYS_PROMPT},
                          {'role': 'user', 'content': chat_history}],
                temperature=0.7, max_tokens=500
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 4:
                print(f'\nFailed: {e}')
                return f'[Generation Failed: {e}]'

def fix_model(model_key, model_id, force_regen=False):
    output_file = f'/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/{model_key}/generated_predictions.jsonl'
    lines = open(input_file).readlines()

    # Load existing results
    existing = {}
    if os.path.exists(output_file) and not force_regen:
        for i, line in enumerate(open(output_file)):
            item = json.loads(line)
            if item.get('predict', '').strip() and not item['predict'].startswith('[Generation Failed'):
                existing[i] = item

    need_regen = [i for i in range(len(lines)) if i not in existing]
    print(f'{model_key}: {len(existing)} valid, {len(need_regen)} to generate')

    if not need_regen:
        print(f'{model_key}: already complete, skipping')
        return

    def process(idx):
        data = json.loads(lines[idx])
        chat_history = parse_llama_factory_prompt(data['prompt'])
        predict = call_api(model_id, chat_history)
        return idx, {'prompt': data['prompt'], 'predict': predict, 'label': data.get('label', '')}

    results = dict(existing)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, i): i for i in need_regen}
        for future in tqdm(as_completed(futures), total=len(need_regen), desc=model_key):
            idx, result = future.result()
            results[idx] = result

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i in range(len(lines)):
            f.write(json.dumps(results[i], ensure_ascii=False) + '\n')

    valid = sum(1 for r in results.values() if r.get('predict','').strip() and not r['predict'].startswith('[Generation Failed'))
    print(f'{model_key}: done, {valid}/{len(lines)} valid')

if __name__ == '__main__':
    # claude: resume (skip existing valid)
    fix_model('claude_sonnet', 'claude-sonnet-4-6', force_regen=False)
    # gpt4o_mini: full regen (brevity contaminated)
    fix_model('gpt4o_mini', 'gpt-4o-mini', force_regen=True)
