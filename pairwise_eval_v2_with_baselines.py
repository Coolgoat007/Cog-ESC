# pairwise_eval_v2_with_baselines.py
# Pairwise评估：Cognitive模型 vs 强基线（V2: Therapeutic Depth）
import json
import random
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import retry, generate

# V2: Therapeutic Depth评估Prompt
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
        if output.strip().endswith("A"):
            winner = "A"
        elif output.strip().endswith("B"):
            winner = "B"

    return winner, output

def parse_llama_factory_prompt(prompt_text):
    try:
        chat_part = prompt_text.split("Conversation:\n")[1]
        chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
        return chat_history_clean
    except:
        return prompt_text

def process_pair(model1_line, model2_line, index, model1_name, model2_name):
    model1_data = json.loads(model1_line)
    model2_data = json.loads(model2_line)

    chat_history = parse_llama_factory_prompt(model1_data['prompt'])
    resp_model1 = model1_data['predict'].strip()
    resp_model2 = model2_data['predict'].strip()

    # 随机打乱 A 和 B
    is_model1_A = random.choice([True, False])
    if is_model1_A:
        resp_A, resp_B = resp_model1, resp_model2
    else:
        resp_A, resp_B = resp_model2, resp_model1

    winner, explanation = get_pairwise_result(chat_history, resp_A, resp_B)

    real_winner = "Tie"
    if winner == "A":
        real_winner = model1_name if is_model1_A else model2_name
    elif winner == "B":
        real_winner = model2_name if is_model1_A else model1_name

    return {
        "index": index,
        "chat_history": chat_history,
        f"response_{model1_name}": resp_model1,
        f"response_{model2_name}": resp_model2,
        f"is_{model1_name}_A": is_model1_A,
        "real_winner": real_winner,
        "explanation_by_judge": explanation
    }

def run_pairwise_eval(model1_file, model2_file, model1_name, model2_name, max_workers=30):
    print(f"\n{'='*60}")
    print(f"Pairwise V2 Evaluation: {model1_name} vs {model2_name}")
    print(f"{'='*60}")

    with open(model1_file, 'r', encoding='utf-8') as f:
        model1_lines = f.readlines()
    with open(model2_file, 'r', encoding='utf-8') as f:
        model2_lines = f.readlines()

    assert len(model1_lines) == len(model2_lines), "Files must have same number of lines!"

    results = {model1_name: 0, model2_name: 0, "Tie": 0}
    total = len(model1_lines)
    detailed_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total):
            futures.append(executor.submit(
                process_pair, model1_lines[i], model2_lines[i], i, model1_name, model2_name
            ))

        for future in tqdm(as_completed(futures), total=total, desc=f"{model1_name} vs {model2_name}"):
            try:
                res = future.result()
                results[res['real_winner']] += 1
                detailed_results.append(res)
            except Exception as e:
                print(f"Error: {e}")

    summary_text = (
        f"\n{'='*60}\n"
        f"Pairwise V2 Results: {model1_name} vs {model2_name} (Total: {total})\n"
        f"{'='*60}\n"
        f"🥇 {model1_name} Wins : {results[model1_name]} ({(results[model1_name]/total)*100:.1f}%)\n"
        f"🥈 {model2_name} Wins : {results[model2_name]} ({(results[model2_name]/total)*100:.1f}%)\n"
        f"🤝 Ties              : {results['Tie']} ({(results['Tie']/total)*100:.1f}%)\n"
        f"{'='*60}\n"
    )
    print(summary_text)

    # 保存结果
    os.makedirs("pairwise_with_baselines_results", exist_ok=True)
    summary_file = f"pairwise_with_baselines_results/{model1_name}_vs_{model2_name}_summary.txt"
    detail_file = f"pairwise_with_baselines_results/{model1_name}_vs_{model2_name}_details.jsonl"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    detailed_results.sort(key=lambda x: x["index"])
    with open(detail_file, 'w', encoding='utf-8') as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Details saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}\n")

    return results

if __name__ == "__main__":
    random.seed(42)

    # 文件路径
    LLAMA_COGNITIVE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_fix/generated_predictions.jsonl"
    QWEN_COGNITIVE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_cognitive_qwens/generated_predictions.jsonl"
    GPT4O = "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o/generated_predictions.jsonl"
    CLAUDE = "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/claude_sonnet/generated_predictions.jsonl"
    GPT4O_MINI = "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines/gpt4o_mini/generated_predictions.jsonl"

    print("\n" + "="*60)
    print("Pairwise V2 Evaluation with Strong Baselines")
    print("="*60)

    all_results = {}

    # 1. Llama-Cognitive vs GPT-4o
    print("\n[1/5] Llama-Cognitive vs GPT-4o")
    all_results["llama_cog_vs_gpt4o"] = run_pairwise_eval(
        LLAMA_COGNITIVE, GPT4O, "Llama-Cognitive", "GPT-4o", max_workers=30
    )

    # 2. Qwen-Cognitive vs GPT-4o
    print("\n[2/5] Qwen-Cognitive vs GPT-4o")
    all_results["qwen_cog_vs_gpt4o"] = run_pairwise_eval(
        QWEN_COGNITIVE, GPT4O, "Qwen-Cognitive", "GPT-4o", max_workers=30
    )

    # 3. Llama-Cognitive vs Claude
    print("\n[3/5] Llama-Cognitive vs Claude")
    all_results["llama_cog_vs_claude"] = run_pairwise_eval(
        LLAMA_COGNITIVE, CLAUDE, "Llama-Cognitive", "Claude-3.5-Sonnet", max_workers=30
    )

    # 4. Qwen-Cognitive vs Claude
    print("\n[4/5] Qwen-Cognitive vs Claude")
    all_results["qwen_cog_vs_claude"] = run_pairwise_eval(
        QWEN_COGNITIVE, CLAUDE, "Qwen-Cognitive", "Claude-3.5-Sonnet", max_workers=30
    )

    # 5. GPT-4o vs Claude (baseline reference)
    print("\n[5/5] GPT-4o vs Claude (Baseline Reference)")
    all_results["gpt4o_vs_claude"] = run_pairwise_eval(
        GPT4O, CLAUDE, "GPT-4o", "Claude-3.5-Sonnet", max_workers=30
    )

    # 打印总结
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for comparison, results in all_results.items():
        print(f"\n{comparison}:")
        for model, count in results.items():
            print(f"  {model}: {count}")
    print("="*60)
