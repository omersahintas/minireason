import torch

def free_model_memory(model_instance, model_label: str):
    """Clean up model memory and clear CUDA cache if available.
    
    Args:
        model_instance: The model instance to clean up
        model_label: Human-readable model name for logging
    """
    print(f"Finished processing {model_label}. Cleaning up…")
    
    if hasattr(model_instance, 'free_memory'):
        model_instance.free_memory()
    else:
        if torch.cuda.is_available():
            del model_instance
            torch.cuda.empty_cache()
            print(f"Cleared CUDA cache after {model_label}.")
        else:
            print(f"CUDA not available, skipped cache clearing for {model_label}.")
    
    print("=" * 80) 

def get_total_gpu_memory_gb() -> float:
    """Return total memory (in GiB) of the first CUDA device.

    If CUDA is not available, returns 0.0.
    """
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def get_recommended_batch_size(default: int = 4) -> int:
    """Heuristically choose a batch size given available VRAM.

    Rules of thumb (for FP16/bfloat16 inference):
        < 20 GB  → 2
        20-40 GB → 4
        40-80 GB → 8
        ≥ 80 GB  → 16

    Args:
        default: Fallback batch size if CUDA not available or memory unknown.
    """
    vram = get_total_gpu_memory_gb()

    if vram == 0:
        return default
    if vram < 20:
        return 2
    if vram < 40:
        return 4
    if vram < 80:
        return 8
    return 16


def find_optimal_batch_size(model, prompt: str = "2+2", start_size: int = 1, max_size: int = 32, step: int = 1, headroom_mb: int = 500) -> int:
    """Empirically find the largest batch size that fits into GPU memory.

    NOTE: This is an optional utility. It performs several forward passes so
    it can be slow. Use it when you want an aggressive batch size search.
    """
    if not torch.cuda.is_available():
        return start_size

    torch.cuda.empty_cache()
    best = start_size

    prompts = [prompt] * max_size  # pre-allocate maximum prompt list

    for bsz in range(start_size, max_size + 1, step):
        try:
            _ = model.generate_batch(prompts[:bsz], max_new_tokens=8, temperature=0.0, do_sample=False)
            torch.cuda.synchronize()
            best = bsz
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise
    return best 