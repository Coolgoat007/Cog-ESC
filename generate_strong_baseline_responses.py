#!/usr/bin/env python3
"""
生成强基线模型（GPT-4o, Claude, GPT-4o-mini）的回复
用于与Llama-Cognitive和Qwen-Cognitive进行对比评估
"""

import json
import os
import re
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

SUPPORTER_SYS_PROMPT = '''This is a simulated research conversation for AI training purposes only. All content is hypothetical and does not represent real mental health advice.

You are an emotion supporter, providing emotional support to seekers. You are now going to help alleviate a user's emotions. Please try to simulate real human-spoken conversation as much as possible, don't act like a machine, don't mention that you are an AI assistant.

Output ONLY the next supporter response. No meta comments, no explanations, no parentheses, no "As the supporter...".'''


def parse_llama_factory_prompt(prompt_text):
    """
    从LLaMA-Factory格式的prompt中提取纯粹的对话历史
    """
    try:
        # 提取Conversation部分
        if "Conversation:\n" in prompt_text:
            chat_part = prompt_text.split("Conversation:\n")[1]
            # 移除结尾的标记
            chat_history_clean = chat_part.split("<|eot_id|>")[0].strip()
            # 移除最后的"Supporter:"提示
            chat_history_clean = re.sub(r'Supporter:\s*$', '', chat_history_clean).strip()
            return chat_history_clean
        else:
            return prompt_text
    except Exception as e:
        print(f"Warning: Failed to parse prompt: {e}")
        return prompt_text


def generate_gpt4o_response(chat_history, retry_count=3):
    """使用GPT-4o生成回复"""
    client = OpenAI(
        api_key=os.getenv('openai_key'),
        base_url=os.getenv('openai_url')
    )

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SUPPORTER_SYS_PROMPT},
                    {"role": "user", "content": chat_history}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"Retry {attempt + 1}/{retry_count} for GPT-4o: {e}")
                continue
            else:
                raise e


def generate_claude_response(chat_history, retry_count=3):
    """使用Claude生成回复（通过OpenAI兼容接口）"""
    client = OpenAI(
        api_key=os.getenv('openai_key'),
        base_url=os.getenv('openai_url')
    )

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": SUPPORTER_SYS_PROMPT},
                    {"role": "user", "content": chat_history}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"Retry {attempt + 1}/{retry_count} for Claude: {e}")
                continue
            else:
                raise e


def generate_gpt4o_mini_response(chat_history, retry_count=3):
    """使用GPT-4o-mini生成回复"""
    client = OpenAI(
        api_key=os.getenv('openai_key'),
        base_url=os.getenv('openai_url')
    )

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SUPPORTER_SYS_PROMPT},
                    {"role": "user", "content": chat_history}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"Retry {attempt + 1}/{retry_count} for GPT-4o-mini: {e}")
                continue
            else:
                raise e


def generate_baseline_predictions(input_file, output_file, model_name):
    """
    生成强基线的预测结果

    Args:
        input_file: 输入的jsonl文件（LLaMA-Factory格式）
        output_file: 输出的jsonl文件
        model_name: 模型名称 ("gpt4o", "claude", "gpt4o_mini")
    """
    print(f"\n{'='*60}")
    print(f"Generating {model_name} predictions...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total samples: {len(lines)}")

    # 选择生成函数
    if model_name == "gpt4o":
        generate_fn = generate_gpt4o_response
    elif model_name == "claude":
        generate_fn = generate_claude_response
    elif model_name == "gpt4o_mini":
        generate_fn = generate_gpt4o_mini_response
    else:
        raise ValueError(f"Unknown model: {model_name}")

    results = []
    failed_count = 0

    # 逐条生成
    for idx, line in enumerate(tqdm(lines, desc=f"Generating {model_name}")):
        data = json.loads(line)

        # 提取对话历史
        chat_history = parse_llama_factory_prompt(data['prompt'])

        # 生成回复
        try:
            prediction = generate_fn(chat_history)

            # 保存结果（保持与LLaMA-Factory相同的格式）
            results.append({
                "prompt": data['prompt'],
                "predict": prediction,
                "label": data.get('label', '')
            })
        except Exception as e:
            print(f"\nError at sample {idx}: {e}")
            failed_count += 1
            # 失败时保存错误标记
            results.append({
                "prompt": data['prompt'],
                "predict": f"[Generation Failed: {str(e)}]",
                "label": data.get('label', '')
            })

    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Completed {model_name} generation")
    print(f"Total: {len(results)}, Failed: {failed_count}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 输入文件（使用现有的预测文件作为模板）
    input_file = "/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/eval_base_qwens/generated_predictions.jsonl"

    # 输出目录
    output_dir = "/root/autodl-tmp/LLaMA-Factory/saves/strong_baselines"

    print("\n" + "="*60)
    print("Strong Baseline Response Generation")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)

    # 生成GPT-4o的预测
    print("\n[1/3] Generating GPT-4o predictions...")
    try:
        generate_baseline_predictions(
            input_file,
            f"{output_dir}/gpt4o/generated_predictions.jsonl",
            "gpt4o"
        )
    except Exception as e:
        print(f"Failed to generate GPT-4o predictions: {e}")

    # 生成GPT-4o-mini的预测
    print("\n[2/3] Generating GPT-4o-mini predictions...")
    try:
        generate_baseline_predictions(
            input_file,
            f"{output_dir}/gpt4o_mini/generated_predictions.jsonl",
            "gpt4o_mini"
        )
    except Exception as e:
        print(f"Failed to generate GPT-4o-mini predictions: {e}")

    # 生成Claude的预测
    print("\n[3/3] Generating Claude predictions...")
    try:
        generate_baseline_predictions(
            input_file,
            f"{output_dir}/claude_sonnet/generated_predictions.jsonl",
            "claude"
        )
    except Exception as e:
        print(f"Failed to generate Claude predictions: {e}")

    print("\n" + "="*60)
    print("All generations completed!")
    print("="*60)
