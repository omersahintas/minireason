from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple


class _MCTSNode:
    """Internal helper node for MCTS search tree."""

    def __init__(self, state: str, parent: Optional["_MCTSNode"] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: List["_MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0  # Average evaluation score in [0,1]
        self.thought: str | None = None  # Thought that led to this node

    # ------------------------------------------------------------------
    # UCB score used for child selection
    # ------------------------------------------------------------------

    def ucb_score(self, c: float = 1.4) -> float:
        if self.visits == 0 or self.parent is None:
            return float("inf")
        exploitation = self.value
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTSReasoner:
    """A minimal Monte Carlo Tree Search reasoner for language models.

    The algorithm:
        1. Selection - traverse the tree using UCB until an unexpanded node is reached.
        2. Expansion - sample *expansion_thoughts* continuations with the model.
        3. Evaluation - ask the model to score the newly expanded state (0-1).
        4. Back-propagation - update visit counts and value estimates.

    Notes:
        • Designed to work with any object that exposes *generate()* (for
          proposing thoughts) and *_evaluate_state()* logic identical to ToT.
        • Accepts *model: Any* to avoid circular import with HFModel.
    """

    def __init__(
        self,
        model: Any,
        num_simulations: int = 30,
        max_depth: int = 3,
        expansion_thoughts: int = 3,
        c_puct: float = 1.4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        evaluation_temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.expansion_thoughts = expansion_thoughts
        self.c_puct = c_puct
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, task_prompt: str) -> Tuple[str, str]:
        root = _MCTSNode(state=task_prompt)

        for _ in range(self.num_simulations):
            node, depth = self._select(root)

            if depth < self.max_depth:
                self._expand(node)
                # pick one newly expanded child (if any) for evaluation
                if node.children:
                    node = random.choice(node.children)
                    depth += 1

            score = self._evaluate_state(task_prompt, node.state)
            self._backpropagate(node, score)

        # Choose the best child of root by highest value
        best_child = max(root.children, key=lambda n: n.value, default=root)
        best_state = best_child.state

        answer_prompt = (
            best_state
            + "\n\nBased on the reasoning above, provide the concise final answer:"
        )
        answer = self.model.generate(answer_prompt, **self.generation_kwargs)
        reasoning = best_state
        return answer.strip(), reasoning.strip()

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: _MCTSNode) -> Tuple[_MCTSNode, int]:
        depth = 0
        while node.children and depth < self.max_depth:
            node = max(node.children, key=lambda n: n.ucb_score(self.c_puct))
            depth += 1
        return node, depth

    def _expand(self, node: _MCTSNode) -> None:
        thoughts = self._propose_thoughts(node.state)
        for t in thoughts:
            child_state = node.state + "\n" + t
            child = _MCTSNode(state=child_state, parent=node)
            child.thought = t
            node.children.append(child)

    def _backpropagate(self, node: _MCTSNode, score: float) -> None:
        while node is not None:
            node.visits += 1
            # incremental mean update
            node.value += (score - node.value) / node.visits
            node = node.parent

    # ------------------------------------------------------------------
    # LM helpers (same as ToT)
    # ------------------------------------------------------------------

    def _propose_thoughts(self, state: str) -> List[str]:
        prompt = (
            state
            + "\n\nList "
            + str(self.expansion_thoughts)
            + " plausible next thoughts, each on its own line starting with a dash '-':"
        )
        raw = self.model.generate(prompt, **self.generation_kwargs)
        lines = [ln.strip() for ln in raw.splitlines()]
        thoughts = [ln.lstrip("- ").strip() for ln in lines if ln.startswith("-")]
        if not thoughts and raw.strip():
            thoughts = [raw.strip()]
        return thoughts[: self.expansion_thoughts]

    def _evaluate_state(self, task_prompt: str, state: str) -> float:
        prompt = (
            f"Task: {task_prompt}\n\nCurrent Reasoning:\n{state}\n\n"
            "Rate how promising this reasoning is on a scale from 0 (worst) to 1 (best). "
            "Only output the score as a decimal number."
        )
        raw = self.model.generate(prompt, **self.eval_kwargs)
        match = re.search(r"\d*\.?\d+", raw)
        if match:
            try:
                score = float(match.group())
            except ValueError:
                score = 0.0
        else:
            score = 0.0
        return max(0.0, min(1.0, score)) 