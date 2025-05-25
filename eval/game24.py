import ast
import operator
from tqdm import tqdm

from typing import Callable, List

import re
from datasets import load_dataset

from models import HFModel
from utils.memory import get_recommended_batch_size

# --------------------------------------------------------
# Safe expression evaluation limited to + - * / parentheses
# --------------------------------------------------------

_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.Num,
    ast.Constant,
    ast.Load,
    ast.UnaryOp,
    ast.Tuple,
}


def _safe_eval(expr: str) -> float | None:
    """Safely evaluate arithmetic expression using ast."""
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    if not all(isinstance(n, tuple(_ALLOWED_NODES)) for n in ast.walk(node)):
        return None
    try:
        result = eval(compile(node, filename="<expr>", mode="eval"), {"__builtins__": {}}, {})
        # Ensure result is a numeric type
        if not isinstance(result, (int, float)):
            return None
        return result
    except Exception:
        return None


# --------------------------------------------------------
# Core evaluation helper
# --------------------------------------------------------

def _evaluate_common(
    model: HFModel,
    dataset_list: list,
    batch_size: int,
    prompt_fn: Callable[[dict], str],
    response_fn: Callable[[List[str]], List[str]],
    desc: str,
) -> tuple[float, int, int]:
    from tqdm import tqdm

    total = correct = 0
    pbar = tqdm(total=len(dataset_list), desc=desc)

    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = response_fn(prompts)

        for item, resp in zip(batch, responses):
            val = _extract_value(resp)
            if val is not None and abs(val - 24) < 1e-3:
                correct += 1
            total += 1
        pbar.update(len(batch))

    pbar.close()
    acc = correct / total if total else 0.0
    return acc, total, correct


# --------------------------------------------------------
# Public API
# --------------------------------------------------------

def evaluate_game24(
    model: HFModel,
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int | None = None,
    max_new_tokens: int = 32,
) -> tuple[float, int, int]:
    """Evaluate model on 24 Game dataset.

    Dataset expected to have fields:
      - "numbers": list[int] (four numbers)

    The prompt asks model to form 24 with arithmetic.
    """
    dataset = load_dataset("nlile/24-game", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset_list = list(dataset)
    if batch_size is None:
        batch_size = get_recommended_batch_size()

    prompt_fn = lambda item: _make_prompt(item["numbers"])

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
        desc="Evaluating Game of 24",
    )


# ------------------------------------------------------------
# pass@k evaluation
# ------------------------------------------------------------


def evaluate_game24_passk(
    model: HFModel,
    k_values: tuple[int, ...] = (1, 5, 10),
    split: str = "train",
    max_samples: int | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Compute pass@k on 24-Game dataset using sampling."""

    dataset = load_dataset("nlile/24-game", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset_list = list(dataset)
    k_max = max(k_values)
    success_counts = {k: 0 for k in k_values}

    for item in tqdm(dataset_list, desc=f"Game24 pass@{k_max}"):
        numbers = item["numbers"]
        prompt = _make_prompt(numbers)
        preds = model.generate_batch(
            [prompt] * k_max,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        for k in k_values:
            hit = False
            for expr in preds[:k]:
                val = _extract_value(expr)
                if val is not None and abs(val - 24) < 1e-3:
                    hit = True
                    break
            if hit:
                success_counts[k] += 1

    total = len(dataset_list)
    return {k: success_counts[k] / total for k in k_values}

def _make_prompt(numbers: list[int]) -> str:
    return (
        f"Use + - * / and parentheses to make 24 from the numbers {numbers}. "
        "Give only the expression:"
    )

def _extract_value(resp: str) -> float | None:
    match = re.search(r"[-+*/() 0-9]+", resp)
    expr_str = match.group(0) if match else resp.strip()
    return _safe_eval(expr_str)