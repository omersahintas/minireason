import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Union


def load_model_and_tokenizer(
    model_name: str, device: Optional[str] = None, model_type: str = "causal", **kwargs
) -> Tuple[Union[AutoModelForCausalLM], AutoTokenizer]:
    """
    Load a model and its corresponding tokenizer from Hugging Face Hub.

    Args:
        model_name (str): Name or path of the model to load
        device (Optional[str]): Device to load the model on ('cuda', 'cpu', etc.)
                               If None, will use CUDA if available, else CPU
        model_type (str): Type of model to load. Currently supports 'causal'
        **kwargs: Additional arguments to pass to the model loading function

    Returns:
        Tuple[Union[AutoModelForCausalLM], AutoTokenizer]: Loaded model and tokenizer

    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If there are issues loading the model or tokenizer
    """
    try:
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model based on type
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move model to specified device
        model = model.to(device)

        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")
