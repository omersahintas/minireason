import torch  # Added for torch.cuda.empty_cache()
from models import HFModel
import argparse
import yaml
from pathlib import Path
import wandb
from utils.memory import free_model_memory, get_recommended_batch_size, find_optimal_batch_size, get_total_gpu_memory_gb
from eval.gsm8k import evaluate_gsm8k, evaluate_gsm8k_cot, evaluate_gsm8k_passk, evaluate_gsm8k_tot
from eval.math500 import evaluate_math500, evaluate_math500_cot, evaluate_math500_tot
from eval.aime2024 import (
    evaluate_aime2024,
    evaluate_aime2024_cot,
    evaluate_aime2024_passk,
    evaluate_aime2024_tot,
)
from eval.game24 import evaluate_game24, evaluate_game24_passk
from eval.crossword5x5 import evaluate_crossword5x5
from eval.gpqa import evaluate_gpqa_diamond
from eval.commonsenseqa import evaluate_commonsenseqa


MODEL_CONFIGS = [
    ("google/gemma-2-2b-it", "Gemma 2B IT"),
    ("google/gemma-2-9b-it", "Gemma 9B IT"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1 Distill Llama 8B"),
    ("gpt2", "GPT-2 (small)"),
    ("EleutherAI/pythia-70m", "Pythia 70M"),
    ("EleutherAI/pythia-2.8b", "Pythia 2.8B"),
]

# Load model-specific generation parameters
_CONFIG_PATH = Path(__file__).parent / "configs" / "model_params.yaml"
try:
    with open(_CONFIG_PATH, "r") as _f:
        MODEL_GEN_PARAMS: dict = yaml.safe_load(_f) or {}
except FileNotFoundError:
    MODEL_GEN_PARAMS = {}

def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks across multiple models.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of HuggingFace model identifiers to evaluate. If omitted, the default MODEL_CONFIGS list defined in the script is used.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per benchmark (use all if omitted).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--find_optimal_batch",
        action="store_true",
        help="Empirically determine the optimal batch size for each model instead of using heuristics.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use (e.g., 0,1,2). Ignored if CUDA unavailable.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["gsm8k", "math500", "aime", "game24", "crossword", "gpqa", "commonsenseqa"],
        help="Select benchmarks to run (default: all). Options: gsm8k, math500, aime, game24, crossword, gpqa, commonsenseqa"
    )
    # Add wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="reasoning-benchmarks",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )

    args = parser.parse_args()

    if args.wandb:
        wandb_config = {
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "find_optimal_batch": args.find_optimal_batch,
            "gpu": args.gpu,
            "benchmarks": args.benchmarks,
            "model_params": MODEL_GEN_PARAMS
        }
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config
        )
    else:
        wandb_run = None

    if args.models:
        MODEL_CONFIGS = [(m, m) for m in args.models]

    all_benchmarks = {
        "gsm8k", "math500", "aime", "game24",
        "crossword", "gpqa", "commonsenseqa"
    }
    selected_benchmarks = set(args.benchmarks) if args.benchmarks else all_benchmarks

    print("Starting evaluation…")
    print("=" * 80)

    # Get batch sizes for each model & benchmark
    vram_key = int(round(get_total_gpu_memory_gb()))
    vram_presets = MODEL_GEN_PARAMS.get("vram_configs", {}).get(vram_key, {})
    preset_batch_sizes = vram_presets.get("batch_sizes", {})

    for model_name, model_label in MODEL_CONFIGS:
        print(f"\n--- Evaluating Model: {model_label} ---")

        device_str = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
        model_instance = HFModel(model_name=model_name, device=device_str)
        print("-" * 50)

        # Retrieve optional generation params for this model
        gen_cfg = MODEL_GEN_PARAMS.get(model_name, {})
        cot_cfg = gen_cfg.get("cot", {})
        tot_cfg = gen_cfg.get("tot", {})
        passk_cfg = gen_cfg.get("passk", {})

        if wandb_run:
            model_config = {
                "model_name": model_name,
                "cot_config": cot_cfg,
                "tot_config": tot_cfg,
                "passk_config": passk_cfg
            }
            
            wandb_run.config.update({f"model_config_{model_name}": model_config}, allow_val_change=True)

        # Preset for given model & benchmark under VRAM key
        def get_batch_size(benchmark_key: str) -> int:
            batch_size = (
                preset_batch_sizes.get(model_name, {}).get(benchmark_key)
                if preset_batch_sizes else 1
            )
            return batch_size

        # GSM8K evaluation
        if "gsm8k" in selected_benchmarks:
            print(f"Evaluating GSM8K ({model_label})...")

            batch_size = get_batch_size("gsm8k")

            accuracy, total, correct = evaluate_gsm8k(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                batch_size=batch_size,
            )
            print(
                f"GSM8K Accuracy for {model_label}: {accuracy:.2%} ({correct}/{total} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "gsm8k/direct/accuracy": accuracy,
                    "gsm8k/direct/correct": correct,
                    "gsm8k/direct/total": total,
                })
                
            print("-" * 50)

            print(f"Evaluating GSM8K with CoT ({model_label})...")
            accuracy_cot, total_cot, correct_cot = evaluate_gsm8k_cot(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                **cot_cfg,
            )
            print(
                f"GSM8K CoT Accuracy for {model_label}: {accuracy_cot:.2%} ({correct_cot}/{total_cot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "gsm8k/cot/accuracy": accuracy_cot,
                    "gsm8k/cot/correct": correct_cot,
                    "gsm8k/cot/total": total_cot,
                    "gsm8k/cot/temperature": cot_cfg.get("temperature", 0.7),
                    "gsm8k/cot/top_p": cot_cfg.get("top_p", 0.9),
                    "gsm8k/cot/max_new_tokens": cot_cfg.get("max_new_tokens", 512),
                })
                
            print("-" * 50)

            print(f"Evaluating GSM8K with ToT ({model_label})...")
            accuracy_tot, total_tot, correct_tot = evaluate_gsm8k_tot(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                **tot_cfg,
            )
            print(
                f"GSM8K ToT Accuracy for {model_label}: {accuracy_tot:.2%} ({correct_tot}/{total_tot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "gsm8k/tot/accuracy": accuracy_tot,
                    "gsm8k/tot/correct": correct_tot,
                    "gsm8k/tot/total": total_tot,
                    "gsm8k/tot/temperature": tot_cfg.get("temperature", 0.7),
                    "gsm8k/tot/top_p": tot_cfg.get("top_p", 0.9),
                    "gsm8k/tot/max_new_tokens": tot_cfg.get("max_new_tokens", 256),
                    "gsm8k/tot/num_thoughts": tot_cfg.get("num_thoughts", 3),
                    "gsm8k/tot/max_depth": tot_cfg.get("max_depth", 3),
                    "gsm8k/tot/k_select": tot_cfg.get("k_select", 2),
                })
                
            print("-" * 50)

            # GSM8K pass@k
            passk = evaluate_gsm8k_passk(
                model_instance,
                split="test",
                k_values=tuple(passk_cfg.get("k_values", (1, 5, 10))),
                max_samples=args.max_samples,
                temperature=passk_cfg.get("temperature", 0.7),
                top_p=passk_cfg.get("top_p", 0.9),
                max_new_tokens=passk_cfg.get("max_new_tokens", 64),
            )
            print(f"GSM8K pass@k: " + ", ".join([f"@{k}={v:.2%}" for k, v in passk.items()]))

            if wandb_run:
                for k, v in passk.items():
                    wandb_run.log({
                        "model": model_label,
                        f"gsm8k/passk/pass@{k}": v,
                        "gsm8k/passk/temperature": passk_cfg.get("temperature", 0.7),
                        "gsm8k/passk/top_p": passk_cfg.get("top_p", 0.9),
                        "gsm8k/passk/max_new_tokens": passk_cfg.get("max_new_tokens", 64),
                    })
                
            print("-" * 50)

        # ------------------------------
        # MATH-500 benchmark (direct)
        # ------------------------------
        if "math500" in selected_benchmarks:
            print(f"\nEvaluating MATH-500 ({model_label})…")

            batch_size = get_batch_size("math500")

            acc_math, total_math, correct_math = evaluate_math500(
                model_instance,
                split="test",
                max_samples=args.max_samples or 500,  # Default to 500 if not specified
                batch_size=batch_size,
            )
            print(
                f"MATH-500 Accuracy for {model_label}: {acc_math:.2%} ({correct_math}/{total_math} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "math500/direct/accuracy": acc_math,
                    "math500/direct/correct": correct_math,
                    "math500/direct/total": total_math,
                })
                
            print("-" * 50)

            print(f"Evaluating MATH-500 with CoT ({model_label})…")
            acc_math_cot, total_math_cot, correct_math_cot = evaluate_math500_cot(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                **cot_cfg,
            )
            print(
                f"MATH-500 CoT Accuracy for {model_label}: {acc_math_cot:.2%} ({correct_math_cot}/{total_math_cot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "math500/cot/accuracy": acc_math_cot,
                    "math500/cot/correct": correct_math_cot,
                    "math500/cot/total": total_math_cot,
                    "math500/cot/temperature": cot_cfg.get("temperature", 0.7),
                    "math500/cot/top_p": cot_cfg.get("top_p", 0.9),
                    "math500/cot/max_new_tokens": cot_cfg.get("max_new_tokens", 512),
                })
                
            print("-" * 50)

            print(f"Evaluating MATH-500 with ToT ({model_label})…")
            acc_math_tot, total_math_tot, correct_math_tot = evaluate_math500_tot(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                **tot_cfg,
            )
            print(
                f"MATH-500 ToT Accuracy for {model_label}: {acc_math_tot:.2%} ({correct_math_tot}/{total_math_tot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "math500/tot/accuracy": acc_math_tot,
                    "math500/tot/correct": correct_math_tot,
                    "math500/tot/total": total_math_tot,
                    "math500/tot/temperature": tot_cfg.get("temperature", 0.7),
                    "math500/tot/top_p": tot_cfg.get("top_p", 0.9),
                    "math500/tot/max_new_tokens": tot_cfg.get("max_new_tokens", 256),
                    "math500/tot/num_thoughts": tot_cfg.get("num_thoughts", 3),
                    "math500/tot/max_depth": tot_cfg.get("max_depth", 3),
                    "math500/tot/k_select": tot_cfg.get("k_select", 2),
                })
                
            print("-" * 50)

        # ------------------------------
        # AIME 2024 benchmark
        # ------------------------------
        if "aime" in selected_benchmarks:
            print(f"Evaluating AIME 2024 ({model_label})…")

            batch_size = get_batch_size("aime")

            acc_aime, total_aime, correct_aime = evaluate_aime2024(
                model_instance,
                split="train",
                max_samples=args.max_samples,
                batch_size=batch_size,
            )
            print(
                f"AIME 2024 Accuracy for {model_label}: {acc_aime:.2%} ({correct_aime}/{total_aime} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "aime/direct/accuracy": acc_aime,
                    "aime/direct/correct": correct_aime,
                    "aime/direct/total": total_aime,
                })
                
            print("-" * 50)

            print(f"Evaluating AIME 2024 with CoT ({model_label})…")
            acc_aime_cot, total_aime_cot, correct_aime_cot = evaluate_aime2024_cot(
                model_instance,
                split="train",
                max_samples=args.max_samples
            )
            print(
                f"AIME 2024 CoT Accuracy for {model_label}: {acc_aime_cot:.2%} ({correct_aime_cot}/{total_aime_cot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "aime/cot/accuracy": acc_aime_cot,
                    "aime/cot/correct": correct_aime_cot,
                    "aime/cot/total": total_aime_cot,
                    "aime/cot/temperature": cot_cfg.get("temperature", 0.7),
                    "aime/cot/top_p": cot_cfg.get("top_p", 0.9),
                    "aime/cot/max_new_tokens": cot_cfg.get("max_new_tokens", 512),
                })
                
            print("-" * 50)

            print(f"Evaluating AIME 2024 with ToT ({model_label})…")
            acc_aime_tot, total_aime_tot, correct_aime_tot = evaluate_aime2024_tot(
                model_instance,
                split="train",
                max_samples=args.max_samples,
                **tot_cfg,
            )
            print(
                f"AIME 2024 ToT Accuracy for {model_label}: {acc_aime_tot:.2%} ({correct_aime_tot}/{total_aime_tot} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "aime/tot/accuracy": acc_aime_tot,
                    "aime/tot/correct": correct_aime_tot,
                    "aime/tot/total": total_aime_tot,
                    "aime/tot/temperature": tot_cfg.get("temperature", 0.7),
                    "aime/tot/top_p": tot_cfg.get("top_p", 0.9),
                    "aime/tot/max_new_tokens": tot_cfg.get("max_new_tokens", 256),
                    "aime/tot/num_thoughts": tot_cfg.get("num_thoughts", 3),
                    "aime/tot/max_depth": tot_cfg.get("max_depth", 3),
                    "aime/tot/k_select": tot_cfg.get("k_select", 2),
                })
                
            print("-" * 50)

            # ------------------ pass@k self-consistency ----------------
            print(f"Computing AIME 2024 pass@k ({model_label})…")

            verifier = HFModel("google/gemma-2-2b-it", device="cpu")
            passk_aime = evaluate_aime2024_passk(
                model_instance,
                k=passk_cfg.get("k", 10),
                max_samples=args.max_samples,
                verifier=verifier,
                temperature=passk_cfg.get("temperature", 0.7),
                top_p=passk_cfg.get("top_p", 0.95),
                max_new_tokens=passk_cfg.get("max_new_tokens", 512),
            )
            print(f"AIME 2024 pass@10: {passk_aime:.2%}")

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "aime/passk/pass@10": passk_aime,
                    "aime/passk/temperature": passk_cfg.get("temperature", 0.7),
                    "aime/passk/top_p": passk_cfg.get("top_p", 0.95),
                    "aime/passk/max_new_tokens": passk_cfg.get("max_new_tokens", 512),
                })
                
            print("-" * 50)

        # ------------------------------
        # Game of 24 benchmark
        # ------------------------------
        if "game24" in selected_benchmarks:
            print(f"Evaluating Game of 24 ({model_label})…")

            batch_size = get_batch_size("game24")

            acc_24, total_24, correct_24 = evaluate_game24(
                model_instance,
                split="train",
                max_samples=args.max_samples,
                batch_size=batch_size,
            )
            print(
                f"Game24 Accuracy for {model_label}: {acc_24:.2%} ({correct_24}/{total_24} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "game24/direct/accuracy": acc_24,
                    "game24/direct/correct": correct_24,
                    "game24/direct/total": total_24,
                })
                
            print("-" * 50)

            passk24 = evaluate_game24_passk(
                model_instance,
                k_values=tuple(passk_cfg.get("k_values", (1, 5, 10))),
                split="train",
                max_samples=args.max_samples,
                temperature=passk_cfg.get("temperature", 0.7),
                top_p=passk_cfg.get("top_p", 0.9),
                max_new_tokens=passk_cfg.get("max_new_tokens", 32),
            )
            print("Game24 pass@k: " + ", ".join([f"@{k}={v:.2%}" for k, v in passk24.items()]))

            if wandb_run:
                for k, v in passk24.items():
                    wandb_run.log({
                        "model": model_label,
                        f"game24/passk/pass@{k}": v,
                        "game24/passk/temperature": passk_cfg.get("temperature", 0.7),
                        "game24/passk/top_p": passk_cfg.get("top_p", 0.9),
                        "game24/passk/max_new_tokens": passk_cfg.get("max_new_tokens", 32),
                    })
                
            print("-" * 50)

        # ------------------------------
        # 5×5 Crossword benchmark
        # ------------------------------
        if "crossword" in selected_benchmarks:
            print(f"Evaluating 5x5 Crossword ({model_label})…")

            batch_size = get_batch_size("crossword")

            acc_cw, total_cw, correct_cw = evaluate_crossword5x5(
                model_instance,
                split="test",
                max_samples=args.max_samples,
                batch_size=batch_size,
            )
            print(
                f"Crossword Accuracy for {model_label}: {acc_cw:.2%} ({correct_cw}/{total_cw} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "crossword/accuracy": acc_cw,
                    "crossword/correct": correct_cw,
                    "crossword/total": total_cw,
                })
                
            print("-" * 50)

        # -------------- External Multiple-Choice tasks -----------------
        if "gpqa" in selected_benchmarks:
            print(f"Evaluating GPQA Diamond ({model_label})…")
            acc_gpqa, total_gpqa, correct_gpqa = evaluate_gpqa_diamond(model_instance)
            print(
                f"GPQA Accuracy for {model_label}: {acc_gpqa:.2%} ({correct_gpqa}/{total_gpqa} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "gpqa/accuracy": acc_gpqa,
                    "gpqa/correct": correct_gpqa,
                    "gpqa/total": total_gpqa,
                })
                
            print("-" * 50)

        if "commonsenseqa" in selected_benchmarks:
            print(f"Evaluating CommonsenseQA ({model_label})…")
            acc_csq, total_csq, correct_csq = evaluate_commonsenseqa(model_instance)
            print(
                f"CommonsenseQA Accuracy for {model_label}: {acc_csq:.2%} ({correct_csq}/{total_csq} correct)"
            )

            if wandb_run:
                wandb_run.log({
                    "model": model_label,
                    "commonsenseqa/accuracy": acc_csq,
                    "commonsenseqa/correct": correct_csq,
                    "commonsenseqa/total": total_csq,
                })
                
            print("-" * 50)

        free_model_memory(model_instance, model_label)

    print("All models evaluated.")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__": 
    main()
