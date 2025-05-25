from datasets import load_dataset
import re
from models import HFModel
from utils.memory import get_recommended_batch_size
from tqdm import tqdm
from typing import Callable, List

# -----------------------------
# Parsing helpers
# -----------------------------

def parse_final_answer(text: str) -> float | None:
    """Parse the model output and return the *final* numeric answer.

    The function looks for a pattern like "Final answer: <number>" (case-insensitive). If
    not present, it falls back to the **last** number in the string. Returns ``None`` when
    no number is found.
    """
    # Try explicit "Final answer:" tag first
    match = re.search(r"final\s+answer[^\d-]*(-?\d+(?:\.\d+)?)", text, re.I)
    if match:
        return float(match.group(1))

    # Fallback ‑ take the *last* number in the output
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return None


def extract_number(text: str) -> float | None:
    """Deprecated helper – kept for backward compatibility."""
    return parse_final_answer(text)

# ---------------------------------
# Internal shared evaluation helper
# ---------------------------------

def _evaluate_common(
    model: HFModel,
    dataset_list: list,
    batch_size: int,
    prompt_fn: Callable[[dict], str],
    response_fn: Callable[[List[str]], List[str]],
    desc: str,
) -> tuple[float, int, int]:
    """Shared evaluation loop used by both direct & CoT modes."""
    total = correct = 0
    pbar = tqdm(total=len(dataset_list), desc=desc)

    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = response_fn(prompts)

        for item, response in zip(batch, responses):
            gold = parse_final_answer(item["answer"].strip())
            pred = parse_final_answer(response)
            if gold is not None and pred is not None and abs(pred - gold) < 1e-3:
                correct += 1
            total += 1
        pbar.update(len(batch))

    pbar.close()
    accuracy = correct / total if total else 0.0
    return accuracy, total, correct

# ---------------------------------
# Direct-answer evaluation (batched)
# ---------------------------------

def evaluate_gsm8k(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int | None = None,
    max_new_tokens: int = 64,
) -> tuple[float, int, int]:
    """Evaluate GSM8K with *direct answer* prompting (no reasoning)."""
    dataset = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if batch_size is None:
        batch_size = get_recommended_batch_size()

    dataset_list = list(dataset)

    prompt_fn = lambda item: f"{item['question'].strip()}\nAnswer:"

    def response_fn(prompts: List[str]) -> List[str]:
        return model.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,
        )

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating GSM8K ({split}, direct)",
    )

# ---------------------------------
# Chain-of-thought evaluation (single-sample)
# ---------------------------------

def evaluate_gsm8k_cot(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[float, int, int]:
    """Evaluate GSM8K using `generate_cot` (single CoT sample per question)."""
    dataset = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    batch_size = 1  # CoT done one example at a time
    dataset_list = list(dataset)

    prompt_fn = lambda item: item["question"].strip()

    def response_fn(prompts: List[str]) -> List[str]:
        # prompts list will have length 1 due to batch_size=1
        return [
            model.generate_cot(
                prompts[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        ]

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating GSM8K ({split}, CoT)",
    )

# ---------------------------------
# Tree-of-Thought evaluation (single-sample)
# ---------------------------------

def evaluate_gsm8k_tot(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = None,
    num_thoughts: int = 3,
    max_depth: int = 3,
    k_select: int = 2,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[float, int, int]:
    """Evaluate GSM8K using Tree-of-Thought reasoning (single ToT sample)."""
    dataset = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if "gpt2" in model.model_name.lower():
        print(f"Skipping GSM8K ToT for {model.model_name} – context window too small")
        return 0.0, 0, 0

    batch_size = 1  # ToT done one example at a time
    dataset_list = list(dataset)

    prompt_fn = lambda item: item["question"].strip()

    def response_fn(prompts: List[str]) -> List[str]:
        return [
            model.generate_tot(
                prompts[0],
                num_thoughts=num_thoughts,
                max_depth=max_depth,
                k_select=k_select,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )[0]  # generate_tot returns (answer, reasoning). We need answer string.
        ]

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating GSM8K ({split}, ToT)",
    )

# ---------------------------------
# pass@k evaluation (sampling)
# ---------------------------------

def evaluate_gsm8k_passk(
    model: HFModel,
    split: str = "test",
    k_values: tuple[int, ...] = (1, 5, 10),
    max_samples: int | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[int, float]:
    """Compute pass@k for GSM8K using stochastic sampling.

    For each question we draw *max(k_values)* independent samples.
    pass@k = fraction of questions for which **any** of the first k samples is correct.
    """
    dataset = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    dataset_list = list(dataset)
    k_max = max(k_values)

    success_counts = {k: 0 for k in k_values}

    for item in tqdm(dataset_list, desc="pass@k GSM8K"):
        prompt = f"{item['question'].strip()}\nAnswer:"
        predictions = []
        for _ in range(k_max):
            pred = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.0,
                do_sample=True,
            )
            predictions.append(pred)
        gold = parse_final_answer(item["answer"].strip())
        for k in k_values:
            if any(
                (
                    (parse_final_answer(p) is not None)
                    and (gold is not None)
                    and abs(parse_final_answer(p) - gold) < 1e-3
                )
                for p in predictions[:k]
            ):
                success_counts[k] += 1

    total = len(dataset_list)
    return {k: success_counts[k] / total for k in k_values} 