"""Token IDs and mask follow HeadTailDataset in the notebook."""


def encode_head_tail(text: str, tokenizer, max_length: int = 256) -> dict[str, list[int]]:
    if max_length != 256:
        raise ValueError("The checkpoint requires max_length=256")
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) <= max_length - 2:
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
        return {"input_ids": list(encoded["input_ids"]), "attention_mask": list(encoded["attention_mask"])}
    half = (max_length - 2) // 2
    ids = [tokenizer.cls_token_id] + tokens[:half] + tokens[-half:] + [tokenizer.sep_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}
