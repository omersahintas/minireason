import re
from typing import Callable, List

from datasets import load_dataset

from models import HFModel
from utils.memory import get_recommended_batch_size

# --------------------------------------------------------
# Helper parse
# --------------------------------------------------------

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _extract_choice(text: str) -> str | None:
    """Return first capital letter A-F in response, else None."""
    match = re.search(r"[A-F]", text.upper())
    if match:
        return match.group(0)
    return None


# --------------------------------------------------------
# Core loop
# --------------------------------------------------------

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
            temperature=1.0,
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


# --------------------------------------------------------
# Shared prompt & answer utilities
# --------------------------------------------------------


def _prompt_fn(item: dict) -> str:
    """Format a GPQA item into a multiple-choice prompt.

    Different dumps of the dataset use slightly different field names
    (e.g. ``"Question"`` vs ``"question"``). We fall back gracefully so
    the evaluation does not crash with a ``KeyError``.
    """

    # Question text may appear under various capitalisations
    q = (
        item.get("question")
        or item.get("Question")
        or item.get("stem")
        or ""
    )

    # Options are consistently stored as a list under the key "options"
    opts = item.get("options", [])

    lines = [f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(opts)]
    return f"{q}\n" + "\n".join(lines) + "\nAnswer (A-F):"


def _answer_fn(item: dict) -> str:
    """Return the gold answer letter for a GPQA item.

    The official dataset stores the solution either as an integer index
    (``answer_index`` / ``answer`` / ``label``) or directly as a letter
    (``solution``). We normalise everything to an integer in ``0-5`` and
    then map to the corresponding A-F label.
    """

    if "answer_index" in item:
        idx = int(item["answer_index"])
    elif "answer" in item:
        idx = int(item["answer"])
    elif "label" in item:
        idx = int(item["label"])
    elif "solution" in item:
        sol = str(item["solution"]).strip().upper()
        idx = _LETTERS.index(sol) if sol in _LETTERS else 0
    else:
        idx = 0  # fallback to first option

    # Clamp to valid range just in case
    idx = max(0, min(idx, len(_LETTERS) - 1))
    return _LETTERS[idx]


# --------------------------------------------------------
# Generic evaluator & thin wrappers per split
# --------------------------------------------------------


def _evaluate_gpqa_split(
    model: HFModel,
    subset: str,
    desc: str,
    *,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> tuple[float, int, int]:
    """Shared evaluation routine for a given GPQA subset (diamond/main/extended)."""

    dataset = load_dataset("Idavidrein/gpqa", subset, split="train")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset_list = list(dataset)
    if batch_size is None:
        batch_size = get_recommended_batch_size()

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=_prompt_fn,
        answer_fn=_answer_fn,
        desc=desc,
    )


# ------------------------
# Public API wrappers
# ------------------------


def evaluate_gpqa_diamond(
    model: HFModel,
    max_samples: int | None = None,
    max_new_tokens: int = 64,  # retained for API compatibility (unused)
    batch_size: int | None = None,
) -> tuple[float, int, int]:
    """Evaluate on the GPQA Diamond subset."""
    return _evaluate_gpqa_split(
        model,
        subset="gpqa_diamond",
        desc="Evaluating GPQA Diamond",
        max_samples=max_samples,
        batch_size=batch_size,
    )


def evaluate_gpqa_main(
    model: HFModel,
    max_samples: int | None = None,
    max_new_tokens: int = 64,  # retained for API compatibility (unused)
    batch_size: int | None = None,
) -> tuple[float, int, int]:
    """Evaluate on the GPQA Main subset."""
    return _evaluate_gpqa_split(
        model,
        subset="gpqa_main",
        desc="Evaluating GPQA Main",
        max_samples=max_samples,
        batch_size=batch_size,
    )


def evaluate_gpqa_extended(
    model: HFModel,
    max_samples: int | None = None,
    max_new_tokens: int = 64,  # retained for API compatibility (unused)
    batch_size: int | None = None,
) -> tuple[float, int, int]:
    """Evaluate on the GPQA Extended subset."""
    return _evaluate_gpqa_split(
        model,
        subset="gpqa_extended",
        desc="Evaluating GPQA Extended",
        max_samples=max_samples,
        batch_size=batch_size,
    ) 