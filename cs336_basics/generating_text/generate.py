import argparse
from pathlib import Path

import torch

from cs336_basics.bpe_tokenizer.tokenizer import BPE_Tokenizer
from cs336_basics.training_loop.checkpointing import load_checkpoint
from cs336_basics.transformer_lm_architecture.softmax import Softmax
from cs336_basics.transformer_lm_architecture.transformer_lm import TransformerLM
from cs336_basics.transformer_lm_training.adamw import AdamW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained language model.")

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eos-token", default="<|endoftext|>")

    return parser.parse_args()


def top_p_sample(probs: torch.Tensor, top_p: float) -> int:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    if top_p < 1.0:
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        keep_mask = cumulative_probs <= top_p
        keep_mask[0] = True

        first_above = torch.nonzero(cumulative_probs >= top_p, as_tuple=False)
        if len(first_above) > 0:
            keep_mask[first_above[0].item()] = True

        sorted_probs = sorted_probs[keep_mask]
        sorted_indices = sorted_indices[keep_mask]

    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled_offset = torch.multinomial(sorted_probs, num_samples=1).item()
    return int(sorted_indices[sampled_offset].item())


def sample_next_token(
    model: TransformerLM,
    input_ids: list[int],
    context_length: int,
    temperature: float,
    top_p: float,
    device: str,
    softmax: Softmax,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    model_input = input_ids[-context_length:]
    tokens = torch.tensor(model_input, dtype=torch.long, device=device).unsqueeze(0)

    logits = model(tokens)[0, -1]
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)

    return top_p_sample(probs, top_p)


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPE_Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("prompt must encode to at least one token.")

    eos_token_ids = tokenizer.encode(eos_token)
    if len(eos_token_ids) != 1:
        raise ValueError("eos_token must map to exactly one token id.")
    eos_token_id = eos_token_ids[0]

    generated_ids = prompt_ids.copy()
    softmax = Softmax(dim=-1)

    model.eval()

    for _ in range(max_new_tokens):
        next_token_id = sample_next_token(
            model=model,
            input_ids=generated_ids,
            context_length=context_length,
            temperature=temperature,
            top_p=top_p,
            device=device,
            softmax=softmax,
        )
        generated_ids.append(next_token_id)
        if next_token_id == eos_token_id:
            break

    return tokenizer.decode(generated_ids)


def build_model(args: argparse.Namespace) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )


def main() -> None:
    args = parse_args()

    tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=[args.eos_token],
    )

    model = build_model(args).to(args.device)
    optimizer = AdamW(model.parameters(), lr=0.0)
    load_checkpoint(args.checkpoint, model, optimizer)

    completion = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token=args.eos_token,
        device=args.device,
    )
    print(completion)


if __name__ == "__main__":
    main()
