import json
import random

V2_FILE = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base/pairwise_eval_v2_details.jsonl"

def extract_best_cases(file_path, num_cases=3):
    cognitive_wins = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['real_winner'] == "Cognitive":
                cognitive_wins.append(data)
                
    # 随机挑选几个 (设定种子保证每次挑的一样)
    random.seed(42)
    selected_cases = random.sample(cognitive_wins, min(num_cases, len(cognitive_wins)))
    
    for idx, case in enumerate(selected_cases):
        print(f"\n{'='*60}")
        print(f"🌟 KILLER CASE {idx + 1} (Index: {case['index']})")
        print(f"{'='*60}")
        print(f"[Context]:\n{case['chat_history']}\n")
        print(f"{'-'*60}")
        print(f"[Base Response (Loser)]:\n{case['response_Base']}\n")
        print(f"{'-'*60}")
        print(f"[Cognitive Response (Winner)]:\n{case['response_Cognitive']}\n")
        print(f"{'-'*60}")
        print(f"[Judge's Explanation]:\n{case['explanation_by_judge']}\n")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    extract_best_cases(V2_FILE)
