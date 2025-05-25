import re
from typing import List, Tuple, Any, Dict, Optional, Union


class TreeOfThoughtReasoner:
    """A lightweight implementation of the Tree-of-Thought (ToT) reasoning framework.

    The algorithm iteratively expands partial chains-of-thought (states),
    evaluates them, and keeps the top-k states at every depth until the
    maximum depth is reached. The final answer is produced from the best
    reasoning path.

    This implementation is designed to work *offline* with any `HFModel`
    instance (e.g. Gemma, Llama) – no OpenAI API required – and keeps the
    interface extremely simple so it can be extended easily later or
    swapped with a fully-fledged package such as
    [`tree-of-thought-llm`](https://github.com/princeton-nlp/tree-of-thought-llm).
    """

    def __init__(
        self,
        model: Any,  # Use Any to avoid circular import with HFModel
        num_thoughts: int = 10,
        max_depth: int = 5,
        k_select: int = 4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        evaluation_temperature: float = 0.7,
    ) -> None:
        """Create a new reasoner.

        Args:
            model: An *already initialised* `HFModel` that will be used both
                for proposing thoughts and for evaluating them.
            num_thoughts: Branching factor (b) – how many thoughts to sample
                from each state.
            max_depth: Maximum depth of the search tree (T in the paper).
            k_select: Beam width (k) – how many top states to keep after each
                evaluation round.
            generation_kwargs: Additional kwargs forwarded to
                `HFModel.generate` (e.g. `max_new_tokens`, `temperature`).
            evaluation_temperature: Temperature used when the model is
                *evaluating* a state. Defaults to 0 for deterministic scoring.
        """
        self.model = model
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.k_select = k_select
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.eval_kwargs = {
            "max_new_tokens": 5,
            "temperature": evaluation_temperature,
            "top_p": 0.0,
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def solve(self, task_prompt: str) -> Tuple[str, str]:
        """Run Tree-of-Thought BFS and return the final answer.

        Args:
            task_prompt: The *original* question/problem statement.

        Returns:
            A tuple `(answer, reasoning)` where `reasoning` is the best chain
            of thought leading to the answer (as a single string with
            newlines).
        """
        # Each state is the *entire* thought string so far (not just the last
        # line). Start with the raw prompt as the only state.
        states: List[str] = [task_prompt]

        for depth in range(self.max_depth):
            candidates = []  # (state_string, score)

            # --------------------------------------------------------------
            # 1) EXPAND: generate `num_thoughts` candidate thoughts for each
            #            current state.
            # --------------------------------------------------------------
            for state in states:
                thoughts = self._propose_thoughts(state)
                for thought in thoughts:
                    new_state = state + "\n" + thought
                    candidates.append(new_state)

            # --------------------------------------------------------------
            # 2) EVALUATE: ask the model to rate each candidate.
            # --------------------------------------------------------------
            scored_candidates = [
                (candidate, self._evaluate_state(task_prompt, candidate))
                for candidate in candidates
            ]

            # --------------------------------------------------------------
            # 3) SELECT: keep top-k states for the next depth.
            # --------------------------------------------------------------
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            states = [cand for cand, _score in scored_candidates[: self.k_select]]

        # ------------------------------------------------------------------
        # 4) ANSWER: build final answer from the best state.
        # ------------------------------------------------------------------
        best_state = states[0]
        answer_prompt = (
            best_state
            + "\n\nBased on the reasoning above, provide the concise final answer:"
        )
        answer = self.model.generate(answer_prompt, **self.generation_kwargs)
        reasoning = best_state
        return answer.strip(), reasoning.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propose_thoughts(self, state: str) -> List[str]:
        """Sample `num_thoughts` continuations for a given reasoning state.
        The model is asked to return each thought on its own line prefixed with '-'.
        Only lines that start with a dash are kept; others are ignored to avoid
        accidentally re-using the original prompt when the model fails to
        comply.
        """
        prompt = (
            state
            + "\n\nAs an expert problem solver, propose "
            + str(self.num_thoughts)
            + " distinct next thoughts to advance the solution. "
            "Return each thought on its own line and prefix it with a dash '-' ."
        )
        raw = self.model.generate(prompt, **self.generation_kwargs)

        candidate_lines = [ln.strip() for ln in raw.splitlines()]
        thoughts = [
            ln.lstrip("- ").strip() for ln in candidate_lines if ln.startswith("-")
        ]

        # Fallback: if the model didn't follow the bullet format, use the whole
        # raw answer as a single thought (to prevent empty expansions).
        if not thoughts and raw.strip():
            thoughts = [raw.strip()]

        # Trim to requested amount
        return thoughts[: self.num_thoughts]

    def _evaluate_state(self, task_prompt: str, state: str) -> float:
        """Ask the model to give a quality score ∈ [0,1] for the state."""
        prompt = (
            f"Task: {task_prompt}\n\nCurrent Reasoning:\n{state}\n\n"
            "Rate how promising this reasoning is on a scale from 0 (worst) to 1 (best)."
            " Only output the score as a decimal number."
        )
        raw_score = self.model.generate(prompt, **self.eval_kwargs)
        # Extract first floating-point number in the output.
        match = re.search(r"\d*\.?\d+", raw_score)
        if match:
            try:
                score = float(match.group())
            except ValueError:
                score = 0.0
        else:
            score = 0.0
        return max(0.0, min(1.0, score))
