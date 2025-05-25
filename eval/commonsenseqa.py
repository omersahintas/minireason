import re
from typing import Callable, List
from datasets import load_dataset

from models import HFModel
from utils.memory import get_recommended_batch_size

_LETTERS = ["A", "B", "C", "D", "E"]


def _extract_choice(text: str) -> str | None:
    m = re.search(r"[A-E]", text.upper())
    if m:
        return m.group(0)
    return None


def _evaluate_common(
    model: HFModel,
    dataset_list: list,
    batch_size: int,
    prompt_fn: Callable[[dict], str],
    answer_fn: Callable[[dict], str],
    desc: str,
):
    from tqdm import tqdm

    total = correct = 0
    pbar = tqdm(total=len(dataset_list), desc=desc)
    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = model.generate_batch(
            prompts,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,
        )
        for item, resp in zip(batch, responses):
            gold = answer_fn(item)
            pred = _extract_choice(resp)
            if pred == gold:
                correct += 1
            total += 1
        pbar.update(len(batch))
    pbar.close()
    return correct / total if total else 0.0, total, correct


def evaluate_commonsenseqa(
    model: HFModel,
    split: str = "validation",
    max_samples: int | None = None,
    batch_size: int | None = None,
):
    """Evaluate CommonsenseQA multiple-choice accuracy."""
    dataset = load_dataset("commonsense_qa", split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    dataset_list = list(dataset)
    if batch_size is None:
        batch_size = get_recommended_batch_size()

    def prompt_fn(item):
        q = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        lines = [f"{label}. {text}" for label, text in zip(labels, choices)]
        prompt = f"{q}\n" + "\n".join(lines) + "\nAnswer (A-E):"
        return prompt

    def answer_fn(item):
        return item["answerKey"].upper()

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        answer_fn=answer_fn,
        desc="Evaluating CommonsenseQA",
    ) 