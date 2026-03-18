import tiktoken


def build_tokenizer():
    """Build a GPT-2 tokenizer extended with custom chat tokens."""
    user_start = "<|user_start|>"
    user_end = "<|user_end|>"
    assistant_start = "<|assistant_start|>"
    assistant_end = "<|assistant_end|>"
    system_start = "<|system_start|>"
    system_end = "<|system_end|>"
    tool_start = "<|tool_start|>"
    tool_end = "<|tool_end|>"
    pad_token = "<|pad|>"

    custom_tokens = [
        pad_token,
        user_start,
        user_end,
        assistant_start,
        assistant_end,
        system_start,
        system_end,
        tool_start,
        tool_end,
    ]

    base = tiktoken.get_encoding("gpt2")
    custom_token_ids = {tok: base.n_vocab + i for i, tok in enumerate(custom_tokens)}

    tokenizer = tiktoken.Encoding(
        name="gpt2_with_custom_tokens",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **custom_token_ids},
    )

    bos_id = tokenizer.eot_token
    bos = tokenizer.decode([bos_id])

    return {
        "tokenizer": tokenizer,
        "bos_id": bos_id,
        "bos": bos,
        "user_start": user_start,
        "user_end": user_end,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "system_start": system_start,
        "system_end": system_end,
        "tool_start": tool_start,
        "tool_end": tool_end,
        "pad_token": pad_token,
        "pad_id": custom_token_ids[pad_token],
        "assistant_start_id": custom_token_ids[assistant_start],
        "assistant_end_id": custom_token_ids[assistant_end],
        "custom_token_ids": custom_token_ids,
        "vocab_size": tokenizer.n_vocab,
    }
