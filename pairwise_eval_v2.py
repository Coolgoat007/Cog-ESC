# pairwise_eval_v2.py
import json
import random
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import retry, generate

# ==========================================
# 盲测 V2：关注“治疗完整度与深度”的 Prompt
# ==========================================
PAIRWISE_PROMPT_V2 = '''You are a senior expert psychologist evaluating emotional support dialogue systems. 
A user (help-seeker) is sharing their emotional distress. Two AI supporters (Assistant A and Assistant B) have generated responses.

Your task is to compare the two responses and decide which one provides better support based on the "Depth and Completeness of Prefrontal Cognitive Regulation".

## Evaluation Criteria (Focus on Therapeutic Quality):
1. Depth of Empathy and Validation: Does the assistant deeply validate the specific underlying emotions, or does it merely provide generic comfort?
2. Structural Completeness: Does the response successfully execute a complete cognitive regulation loop (i.e., First deeply acknowledging the emotion, THEN gently offering reframing, insights, or constructive strategies)?
3. Constructiveness of Strategy: Are the offered perspectives or strategies actually helpful, cognitively sound, and tailored to the user's situation?
4. **CRITICAL NOTE ON LENGTH AND PACING**: Do not penalize a response simply for being detailed, structured, or longer. A comprehensive, multi-step therapeutic response is highly encouraged and considered superior, provided it does not skip the initial emotional validation phase.

## Conversation Context:
{chat_history}

## Assistant A:
{response_A}

## Assistant B:
{response_B}

## Evaluation Steps:
1. Analyze Assistant A's response. Did it provide deep validation? Was its strategy constructive?
2. Analyze Assistant B's response using the same criteria.
3. Decide which assistant provided a more complete and professionally sound cognitive support response. If both are equally good or bad, choose Tie.

## Output Format:
You must output your final decision on the last line exactly as follows:
Winner: A
or
Winner: B
or
Winner: Tie

Provide a brief reason before your final decision.
'''

@retry()
def get_pairwise_result(chat_history, resp_A, resp_B):
    query = PAIRWISE_PROMPT_V2.format(
        chat_history=chat_history, 
        response_A=resp_A, 
        response_B=resp_B
    )
    output = generate(query)
    
    # 解析结果
    winner = "Tie"
    if "Winner: A" in output:
        winner = "A"
    elif "Winner: B" in output:
        winner = "B"
    elif "Winner: Tie" in output:
        winner = "Tie"
    else:
        if output.strip().endswith("A"): winner = "A"
        elif output.strip().endswith("B"): winner = "B"
        
    return winner, output

def parse_llama_factory_prompt(prompt_text):
    try:
        chat_part = prompt_text.split("Conversation:\n")[1]
        chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
        return chat_history_clean
    except:
        return prompt_text

def process_pair(base_line, cog_line, index):
    base_data = json.loads(base_line)
    cog_data = json.loads(cog_line)
    
    chat_history = parse_llama_factory_prompt(base_data['prompt'])
    resp_base = base_data['predict'].strip()
    resp_cog = cog_data['predict'].strip()
    
    # 随机打乱 A 和 B
    is_base_A = random.choice([True, False])
    if is_base_A:
        resp_A, resp_B = resp_base, resp_cog
    else:
        resp_A, resp_B = resp_cog, resp_base
        
    winner, explanation = get_pairwise_result(chat_history, resp_A, resp_B)
    
    real_winner = "Tie"
    if winner == "A":
        real_winner = "Base" if is_base_A else "Cognitive"
    elif winner == "B":
        real_winner = "Cognitive" if is_base_A else "Base"
        
    return {
        "index": index,
        "chat_history": chat_history,
        "response_Base": resp_base,
        "response_Cognitive": resp_cog,
        "is_base_A": is_base_A,
        "real_winner": real_winner,
        "explanation_by_judge": explanation
    }

def run_pairwise_eval(base_file, cog_file, max_workers=15):
    print(f"Loading data with {max_workers} threads...")
    with open(base_file, 'r', encoding='utf-8') as f:
        base_lines = f.readlines()
    with open(cog_file, 'r', encoding='utf-8') as f:
        cog_lines = f.readlines()
        
    assert len(base_lines) == len(cog_lines), "Error: The two files must have the same number of lines!"
    
    results = {"Cognitive": 0, "Base": 0, "Tie": 0}
    total = len(base_lines)
    detailed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total):
            futures.append(executor.submit(process_pair, base_lines[i], cog_lines[i], i))
            
        for future in tqdm(as_completed(futures), total=total, desc="Pairwise V2 Judging"):
            try:
                res = future.result()
                results[res['real_winner']] += 1
                detailed_results.append(res)
            except Exception as e:
                print(f"Error processing a pair: {e}")
                
    summary_text = (
        f"\n{'='*50}\n"
        f"Pairwise V2 (Therapeutic Depth) Results (Total: {total})\n"
        f"{'='*50}\n"
        f"🥇 Cognitive Wins : {results['Cognitive']} ({(results['Cognitive']/total)*100:.1f}%)\n"
        f"🥈 Base Wins      : {results['Base']} ({(results['Base']/total)*100:.1f}%)\n"
        f"🤝 Ties           : {results['Tie']} ({(results['Tie']/total)*100:.1f}%)\n"
        f"{'='*50}\n"
    )
    print(summary_text)
    
    # 存入带 _v2 后缀的新文件，保护你之前的实验数据
    base_dir = os.path.dirname(base_file)
    summary_file = os.path.join(base_dir, "pairwise_eval_v2_summary_qwen_cognitive_ablation.txt")
    detail_file = os.path.join(base_dir, "pairwise_eval_v2_details_qwen_cognitive_ablation.jsonl")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    detailed_results.sort(key=lambda x: x["index"])
    with open(detail_file, 'w', encoding='utf-8') as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"V2 Details saved to: {detail_file}")
    print(f"V2 Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    BASE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl"
    COGNITIVE_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions.jsonl"
    ABLATION_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_ablation_qwen/generated_predictions.jsonl"
    
    random.seed(42)
    print("Starting LLM-as-a-Judge Pairwise V2 Blind Test (Focus on Depth)...")
    # run_pairwise_eval(BASE_FILE, COGNITIVE_FILE, max_workers=30) # 如果之前 30 线程没报错，这里可以调到 20-30
    run_pairwise_eval(ABLATION_FILE, COGNITIVE_FILE, max_workers=30)
