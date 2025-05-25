import re
from typing import Callable, List, Optional

from datasets import load_dataset

from models import HFModel
from utils.memory import get_recommended_batch_size

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def _extract_boxed_answer(text) -> str | None:
    """Extract AIME answer inside \boxed{...} pattern or last integer.

    Returns None if nothing can be parsed.
    """
    # Convert non-string inputs (e.g. ints) to string first
    if not isinstance(text, str):
        text = str(text)

    boxed = re.search(r"\\boxed\s*{([^}]+)}", text)
    if boxed:
        return boxed.group(1).strip()

    # Fallback – last integer in text
    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1]
    return None


# --------------------------------------------------------
# Shared dataset loading & filtering
# --------------------------------------------------------

def _load_aime_dataset(split: str, max_samples: int | None) -> list:
    """Load AIME-2024 problems from dedicated dataset or MATH benchmark."""
    try:
        dataset_dict = load_dataset("Maxwell-Jia/AIME_2024")
        dataset = dataset_dict[split]
        dedicated = True
    except Exception:
        dataset_dict = load_dataset("nlile/hendrycks-MATH-benchmark")
        dataset = dataset_dict.get(split) or dataset_dict[list(dataset_dict.keys())[0]]
        dedicated = False

    if not dedicated:
        def _is_aime_2024(item):
            meta = item.get("metadata") or ""
            txt = (meta + " " + _get_problem_text(item)).lower()
            return ("aime" in txt) and ("2024" in txt)
        
        dataset = dataset.filter(_is_aime_2024)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return list(dataset)

def _get_problem_text(item: dict) -> str:
    """Get problem text across dataset versions."""
    return (item.get("problem") or item.get("Problem", "")).strip()

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

    def _get_gold(item: dict) -> str:
        for key in ("answer", "Answer", "solution", "Solution"):
            if key in item and item[key] is not None:
                return str(item[key]).strip()
        return ""

    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = response_fn(prompts)

        for item, resp in zip(batch, responses):
            gold_raw = _get_gold(item)
            gold = _extract_boxed_answer(gold_raw)
            pred = _extract_boxed_answer(resp)
            if gold is not None and pred is not None and str(gold) == str(pred):
                correct += 1
            total += 1
        pbar.update(len(batch))

    pbar.close()
    acc = correct / total if total else 0.0
    return acc, total, correct

# --------------------------------------------------------
# Updated evaluation functions (now using shared logic)
# --------------------------------------------------------

def evaluate_aime2024(
    model: HFModel,
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int | None = None,
    max_new_tokens: int = 64,
) -> tuple[float, int, int]:
    dataset_list = _load_aime_dataset(split, max_samples)
    
    if batch_size is None:
        batch_size = get_recommended_batch_size()

    prompt_fn = lambda item: _get_problem_text(item) + "\nProvide only the final answer enclosed in \\boxed{ }."

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
        desc="Evaluating AIME 2024",
    )


# --------------------------------------------------------
# CoT evaluation
# --------------------------------------------------------

def evaluate_aime2024_cot(
    model: HFModel,
    split: str = "train",
    max_samples: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> tuple[float, int, int]:
    """Chain-of-thought evaluation for 2024 AIME problems."""

    dataset_list = _load_aime_dataset(split, max_samples)

    # CoT generation produces long reasoning; keep batch_size=1 to avoid OOM
    batch_size = 1

    prompt_fn = lambda item: _get_problem_text(item)

    def response_fn(prompts: List[str]) -> List[str]:
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
        desc="Evaluating AIME 2024 CoT",
    )


# --------------------------------------------------------
# pass@k self-consistency + optional verifier
# --------------------------------------------------------

def _llm_verify_equivalence(
    verifier: HFModel,
    problem: str,
    gold: str,
    pred: str,
) -> bool:
    """Ask *verifier* whether *pred* equals *gold* for a problem.

    The verifier is expected to answer with a single token "YES" or "NO".
    We fall back to string match if the output is ambiguous.
    """

    v_prompt = (
        f"Problem: {problem}\n"
        f"Gold answer: {gold}\n"
        f"Student answer: {pred}\n"
        "Are the gold answer and student answer exactly equivalent?"
        " Respond with YES or NO only."
    )

    try:
        resp = verifier.generate(
            v_prompt,
            max_new_tokens=4,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
    except Exception:
        return False  # Any error -> treat as incorrect

    resp = resp.strip().lower()
    if resp.startswith("yes"):
        return True
    if resp.startswith("no"):
        return False
    # Fallback – string equality as last resort
    return str(gold).strip() == str(pred).strip()


def evaluate_aime2024_passk(
    model: HFModel,
    k: int = 5,
    *,
    max_samples: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verifier: Optional[HFModel] = None,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.1,
):
    """Self-consistency evaluation (pass@k) with optional LLM verifier."""

    from tqdm import tqdm

    dataset_list = _load_aime_dataset(split="train", max_samples=max_samples)

    success = 0
    for item in tqdm(dataset_list, desc=f"AIME pass@{k}"):
        problem_txt = _get_problem_text(item)
        prompt = problem_txt + "\nProvide only the final answer enclosed in \\boxed{ }."

        # Draw k independent samples in a single batched call for speed
        raw_samples = model.generate_batch(
            [prompt] * k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )
        preds = [_extract_boxed_answer(r) or r.strip() for r in raw_samples]

        gold = _extract_boxed_answer(
            item.get("Answer", "")
            or item.get("answer", "")
            or item.get("solution", "")
            or item.get("Solution", "")
        )

        print("Gold: \n", gold)

        hit = False
        for p in preds:

            if gold is None:
                break  # no answer – should not happen
            if verifier is None:
                # Exact match fallback
                if str(p).strip() == str(gold).strip():
                    hit = True
                    break
            else:
                if _llm_verify_equivalence(verifier, problem_txt, gold, p):
                    hit = True
                    break

        if hit:
            success += 1

    total = len(dataset_list)
    return success / total if total else 0.0 

# --------------------------------------------------------
# Tree-of-Thought evaluation
# --------------------------------------------------------

def evaluate_aime2024_tot(
    model: HFModel,
    split: str = "train",
    max_samples: int | None = None,
    num_thoughts: int = 3,
    max_depth: int = 3,
    k_select: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> tuple[float, int, int]:
    """Evaluate AIME-2024 problems using Tree-of-Thought reasoning.

    The evaluation is identical to the CoT variant except that each problem is
    solved via ``HFModel.generate_tot`` (single sample).
    """
    # GPT-2 context window is too small for iterative ToT prompts
    if "gpt2" in model.model_name.lower():
        print(f"Skipping AIME-2024 ToT for {model.model_name} – context window too small")
        return 0.0, 0, 0

    dataset_list = _load_aime_dataset(split, max_samples)
    batch_size = 1  # ToT is iterative and memory heavy

    prompt_fn = lambda item: _get_problem_text(item).strip()

    def response_fn(prompts: List[str]) -> List[str]:
        answer, _reasoning = model.generate_tot(
            prompts[0],
            num_thoughts=num_thoughts,
            max_depth=max_depth,
            k_select=k_select,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return [answer]

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc="Evaluating AIME 2024 ToT",
    ) 