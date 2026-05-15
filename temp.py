import json
from pathlib import Path


ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
    "tool": "observation",
    "function": "function_call",
}


def normalize_message(msg: dict) -> dict | None:
    role = msg.get("role")
    content = msg.get("content")

    if role is None or content is None:
        return None

    role = role.strip().lower()
    mapped_role = ROLE_MAP.get(role)
    if mapped_role is None:
        return None

    if not isinstance(content, str):
        content = str(content)

    return {"from": mapped_role, "value": content}


def normalize_reply(reply: dict | str | None) -> dict | None:
    if reply is None:
        return None

    if isinstance(reply, str):
        text = reply
        role = "gpt"
    elif isinstance(reply, dict):
        text = reply.get("content")
        role = ROLE_MAP.get(str(reply.get("role", "assistant")).strip().lower(), "gpt")
    else:
        return None

    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    return {"from": role, "value": text}


def convert_sample(sample: dict) -> dict | None:
    messages = sample.get("messages")
    chosen = sample.get("chosen")
    rejected = sample.get("reject") if "reject" in sample else sample.get("rejected")

    if not isinstance(messages, list) or len(messages) == 0:
        return None

    chosen_norm = normalize_reply(chosen)
    rejected_norm = normalize_reply(rejected)

    if chosen_norm is None or rejected_norm is None:
        return None

    if chosen_norm["value"].strip() == "" or rejected_norm["value"].strip() == "":
        return None

    if chosen_norm["value"].strip() == rejected_norm["value"].strip():
        return None

    system_prompt = None
    conversations = []

    for i, msg in enumerate(messages):
        norm = normalize_message(msg)
        if norm is None:
            return None

        if i == 0 and norm["from"] == "system":
            system_prompt = norm["value"]
        else:
            # ShareGPT preference format expects conversations without the final assistant answer,
            # and chosen/rejected as the competing assistant replies.
            if norm["from"] == "system":
                # system only keep the first one separately
                continue
            conversations.append(norm)

    if len(conversations) == 0:
        return None

    # 基本合法性：conversations 最后最好是 human
    # 这样 chosen/rejected 才是“下一条 assistant 回复”
    if conversations[-1]["from"] != "human":
        return None

    output = {
        "conversations": conversations,
        "chosen": chosen_norm,
        "rejected": rejected_norm,
    }

    if system_prompt:
        output["system"] = system_prompt

    # 额外保留元信息，不影响 LLaMA-Factory 使用
    if "scene" in sample:
        output["scene"] = sample["scene"]
    if "description" in sample:
        output["description"] = sample["description"]

    return output


def main():
    # input_path = Path("./output/data_base/0-60_max_Q_cmp_dpo.json")
    input_path = Path("./output/data_ablation_no_validation_first/0-60_max_Q_cmp_dpo.json")
    # input_path = Path("./output/data_ablation_prompt/0-60_max_Q_cmp_dpo.json")
    # input_path = Path("./output/data_ablation_strategy/0-60_max_Q_cmp_dpo.json")
    # output_path = Path("ESC-Pro-base.json")
    output_path = Path("ESC-Pro-ablation_no_validation_first.json")
    # output_path = Path("ESC-Pro-ablation-prompt.json")
    # output_path = Path("ESC-Pro-ablation-strategy.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    dropped = 0

    for sample in data:
        new_sample = convert_sample(sample)
        if new_sample is None:
            dropped += 1
            continue
        converted.append(new_sample)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"input samples   : {len(data)}")
    print(f"converted       : {len(converted)}")
    print(f"dropped invalid : {dropped}")
    print(f"saved to        : {output_path}")


if __name__ == "__main__":
    main()
