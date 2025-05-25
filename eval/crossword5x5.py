from typing import Callable, List
import textwrap

from datasets import load_dataset

from models import HFModel
from utils.memory import get_recommended_batch_size

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _normalize(grid_str: str) -> str:
    """Canonicalise grid by stripping whitespace and upper-casing."""
    return "".join(grid_str.split()).upper()


# --------------------------------------------------------
# Core loop
# --------------------------------------------------------

def _evaluate_common(
    model: HFModel,
    dataset_list: list,
    batch_size: int,
    prompt_fn: Callable[[dict], str],
    response_fn: Callable[[List[str]], List[str]],
    desc: str,
):
    from tqdm import tqdm

    total = correct = 0
    pbar = tqdm(total=len(dataset_list), desc=desc)

    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = response_fn(prompts)

        for item, resp in zip(batch, responses):
            gold = _normalize(item["solution"])
            pred = _normalize(resp)
            if pred == gold:
                correct += 1
            total += 1
        pbar.update(len(batch))

    pbar.close()
    return correct / total if total else 0.0, total, correct


# --------------------------------------------------------
# Public API
# --------------------------------------------------------

def evaluate_crossword5x5(
    model: HFModel,
    dataset_name: str = "taishiisun/5x5_crossword",
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int | None = None,
    max_new_tokens: int = 64,
):
    """Evaluate model on 5x5 crossword fill-in task.

    Dataset is expected to have fields:
      • "clues": string describing across/down clues
      • "solution": string containing filled grid (25 letters, row-major)
    """
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset_list = list(dataset)
    if batch_size is None:
        batch_size = get_recommended_batch_size()

    def prompt_fn(item):
        instruction = textwrap.dedent(
            f"""
            Fill in the following 5x5 crossword grid. Respond with the 25 letters row by row with no spaces or newlines.
            Clues:
            {item['clues']}
            Grid:
            {item.get('grid','')}
            Answer (25 letters):"""
        ).strip()
        return instruction

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
        desc="Evaluating 5x5 Crossword",
    ) 