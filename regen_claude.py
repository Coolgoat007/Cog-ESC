import os, json
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, '.')
from generate_strong_baseline_responses import SUPPORTER_SYS_PROMPT, parse_llama_factory_prompt

client = OpenAI(api_key=os.getenv('openai_key'), base_url=os.getenv('openai_url'))

input_file = '/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl'
output_file = '/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/claude_sonnet/generated_predictions.jsonl'

with open(input_file, 'r') as f:
    lines = f.readlines()

def process(args):
    idx, line = args
    data = json.loads(line)
    chat_history = parse_llama_factory_prompt(data['prompt'])
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model='claude-sonnet-4-6',
                messages=[{'role': 'system', 'content': SUPPORTER_SYS_PROMPT}, {'role': 'user', 'content': chat_history}],
                temperature=0.7, max_tokens=500
            )
            return idx, {'prompt': data['prompt'], 'predict': r.choices[0].message.content.strip(), 'label': data.get('label', '')}
        except Exception as e:
            if attempt == 2:
                print(f'\nFailed idx {idx}: {e}')
                return idx, {'prompt': data['prompt'], 'predict': '', 'label': data.get('label', '')}

results = [None] * len(lines)
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process, (i, line)): i for i, line in enumerate(lines)}
    for future in tqdm(as_completed(futures), total=len(lines), desc='Claude'):
        idx, result = future.result()
        results[idx] = result

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Done: {len(results)} samples')
