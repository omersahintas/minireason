import torch
import logging
import re
from utils.loader import load_model_and_tokenizer
from .tot import TreeOfThoughtReasoner
from .mcts import MCTSReasoner


# Models known to not have chat templates
NON_CHAT_MODELS = [
    "gpt2", 
    "EleutherAI/pythia-2.8b"
]


class HFModel:
    def __init__(self, model_name="google/gemma-2-2b-it", device=None):
        """
        Initialize the Gemma model.

        Args:
            model_name (str): The model name/path to load
            device (str, optional): Device to place model on ('cuda', 'cpu', etc.).
                                    If None, will use CUDA if available.
        """
        self.model_name = model_name

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")

        # Set default dtype based on device
        kwargs = {}
        if self.device == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32

        # Load model and tokenizer using the generic loader
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_name, device=self.device, model_type="causal", **kwargs
        )

        # Ensure correct padding side for decoder-only LMs (e.g., GPT-2)
        if not getattr(self.model.config, "is_encoder_decoder", False):
            # decoder-only – use left padding so that last tokens align
            self.tokenizer.padding_side = "left"
            # some tokenizers need pad_token to exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Handle models without pad token (like GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set padding token to EOS token for {model_name}")

        self.model.eval()
        print(f"Model loaded successfully!")

    def _build_messages(self, user_content: str):
        """Return a list of chat messages in the format expected by apply_chat_template."""
        return [{"role": "user", "content": user_content}]

    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=None,
    ):
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The input prompt to generate from
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness in generation
            top_p (float): Nucleus sampling parameter

        Returns:
            str: The generated text
        """

        # Build chat template prompt via tokenizer helper
        messages = self._build_messages(prompt)
        formatted_prompt: str
        
        # Check if model is in the list of known non-chat models
        if any(model_id in self.model_name for model_id in NON_CHAT_MODELS):
            formatted_prompt = prompt
        elif hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, KeyError) as e:
                logging.warning(
                    "Chat template fallback for %s: %s - using raw prompt",
                    self.model_name, str(e)
                )
                # Tokenizer does not define a chat template (e.g., GPT-2). Use raw prompt.
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample if do_sample is not None else (temperature != 0),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt portion
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return response

    def generate_cot(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.95):
        """
        Generate chain-of-thought reasoning.

        Args:
            prompt (str): The input prompt for CoT reasoning
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness in generation
            top_p (float): Nucleus sampling parameter

        Returns:
            str: The generated reasoning
        """
        cot_prompt = (
            f"{prompt}\n\n"
            "Let's think step by step. "
            "When you are done, output only the final answer enclosed in \\boxed{ }."
        )

        raw =  self.generate(
            cot_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
        )
        m = re.search(r"\\boxed\{(.+?)\}", raw, flags=re.DOTALL)
        return m.group(1).strip() if m else raw

    def free_memory(self):
        """
        Completely free GPU memory by deleting model and tokenizer objects and emptying CUDA cache.
        Call this method when you're done with the model to prevent memory leaks.
        """
        if hasattr(self, "model"):
            del self.model
            self.model = None

        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory completely freed.")

    # ------------------------------------------------------------------
    # Tree-of-Thought Reasoning
    # ------------------------------------------------------------------

    def generate_tot(
        self,
        prompt: str,
        num_thoughts: int = 3,
        max_depth: int = 3,
        k_select: int = 2,
        **generation_kwargs,
    ):
        """Solve a task using the Tree-of-Thought (ToT) framework.

        This is a thin wrapper around `TreeOfThoughtReasoner` so that users can
        call the algorithm directly from an `HFModel` instance without having
        to import the reasoner class themselves.

        Args:
            prompt: The task/question to solve.
            num_thoughts: Branching factor (b).
            max_depth: Maximum search depth (T).
            k_select: Beam width (k).
            **generation_kwargs: Extra kwargs forwarded to the underlying
                `HFModel.generate` call used by the reasoner (e.g.
                `max_new_tokens`, `temperature`).

        Returns:
            Tuple `(answer, reasoning)` where `answer` is the final solution
            and `reasoning` is the best chain-of-thought discovered.
        """
        reasoner = TreeOfThoughtReasoner(
            model=self,
            num_thoughts=num_thoughts,
            max_depth=max_depth,
            k_select=k_select,
            generation_kwargs=generation_kwargs or None,
        )
        return reasoner.solve(prompt)

    # ------------------------------------------------------------------
    # Monte Carlo Tree Search Reasoning
    # ------------------------------------------------------------------

    def generate_mcts(
        self,
        prompt: str,
        num_simulations: int = 30,
        max_depth: int = 3,
        expansion_thoughts: int = 3,
        evaluation_temperature: float = 0.7,
        **generation_kwargs,
    ):
        """Solve a task using MCTS reasoning framework.

        Args:
            prompt: Task/question statement.
            num_simulations: Number of MCTS simulations (rollouts).
            max_depth: Maximum reasoning depth.
            expansion_thoughts: Number of candidate thoughts per expansion.
            evaluation_temperature: Temperature for evaluation
            **generation_kwargs: Forwarded to underlying generate calls.
        """
        reasoner = MCTSReasoner(
            model=self,
            num_simulations=num_simulations,
            max_depth=max_depth,
            expansion_thoughts=expansion_thoughts,
            generation_kwargs=generation_kwargs or None,
            evaluation_temperature=evaluation_temperature,
        )
        return reasoner.solve(prompt)

    def generate_batch(
        self,
        prompts,
        max_new_tokens=256,
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=False,
    ):
        """Generate a list of responses given a list of *prompts*.

        Args:
            prompts (List[str]): Prompts for batch generation.
            Other args mirror those of ``generate``.
        Returns:
            List[str]: Decoded responses for each prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Build chat-formatted prompts
        formatted_prompts = []
        for p in prompts:
            # Check if model is in the list of known non-chat models
            if any(model_id in self.model_name for model_id in NON_CHAT_MODELS):
                formatted = p
            elif hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    messages = self._build_messages(p)
                    formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except (ValueError, KeyError):
                    formatted = p
            else:
                formatted = p
            formatted_prompts.append(formatted)

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample if do_sample is not None else (temperature != 0),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded_list = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = []
        for fmt_prompt, full_text in zip(formatted_prompts, decoded_list):
            responses.append(full_text.replace(fmt_prompt, "").strip())
        return responses
