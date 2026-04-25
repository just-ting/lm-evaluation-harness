from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


CODECS = [
    "fp16",
    "int8",
    "int4",
    "residual_int8",
    "residual_int4",
    "residual_int8_lz4",
    "ds_int8",
    "ds_int4",
    "ds_residual_int8",
    "ds_residual_int4",
    "ds_outlier_int4",
    "ds_residual_outlier_int4",
    "ds_residual_int8_lz4",
    "outlier_int2",
    "outlier_int1",
]

TASK_METRICS = {
    "piqa": ["acc", "acc_norm"],
    "lambada_openai": ["acc", "perplexity"],
    "coqa": ["em", "f1"],
    "gsm8k": ["exact_match"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hs_codec_matrix_runs"),
    )
    parser.add_argument("--task", default="piqa")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--hs-max-gen-len",
        type=int,
        default=None,
        help="Optional HS staged past-hidden sequence length override. If omitted, the adapter default is used.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Pass --log_samples/--output_path through to lm_eval and compare per-sample outputs.",
    )
    parser.add_argument(
        "--sample-field",
        default="filtered_resps",
        choices=["filtered_resps", "resps"],
        help="Sample field to compare against the fp16 baseline when --log-samples is set.",
    )
    parser.add_argument(
        "--codecs",
        nargs="*",
        default=CODECS,
    )
    return parser.parse_args()


def metrics_for_task(task: str) -> list[str]:
    return TASK_METRICS.get(
        task,
        ["acc", "acc_norm", "perplexity", "em", "f1", "exact_match"],
    )


def parse_metric(output: str, task: str, metric: str) -> str:
    pattern = re.compile(
        rf"^\|{re.escape(task)}\s*\|.*\|{re.escape(metric)}\s*\|[^|]*\|([^|]+)\|",
        re.MULTILINE,
    )
    match = pattern.search(output)
    if not match:
        return ""
    return match.group(1).strip()


def parse_metric_from_results_json(results_path: Path, task: str, metric: str) -> str:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    task_results = payload.get("results", {}).get(task, {})
    for key, value in task_results.items():
        if key == metric or key.startswith(f"{metric},"):
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
    return ""


def summarize_failure(output: str) -> str:
    patterns = [
        r"Traceback \(most recent call last\):.*",
        r"ModuleNotFoundError:.*",
        r"ImportError:.*",
        r"RuntimeError:.*",
        r"ValueError:.*",
        r"AssertionError:.*",
        r"NotImplementedError:.*",
        r"ERROR .*",
        r"\[rank0\]:.*",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            text = match.group(0).strip()
            if len(text) > 600:
                text = text[:600].rstrip() + "..."
            return text.replace("\n", " ")
    text = output.strip()
    if len(text) > 600:
        text = text[:600].rstrip() + "..."
    return text


def determine_notes(output: str, task: str, status: str) -> str:
    if status == "success":
        if "Running loglikelihood requests" in output:
            return f"Completed formal lm_eval loglikelihood flow for {task}."
        if "Running generate_until requests" in output:
            return f"Completed formal lm_eval generate_until flow for {task}."
        return f"Completed formal lm_eval flow for {task}."
    summary = summarize_failure(output)
    lower = output.lower()
    if "unknown codec" in lower:
        return f"{summary} Root cause: codec registry rejection."
    if "importerror" in lower or "modulenotfounderror" in lower:
        return f"{summary} Root cause: missing dependency during codec/runtime initialization."
    if "set_hidden_state_offload" in lower or "hs_offload" in lower:
        return f"{summary} Root cause: HS offload initialization/runtime failure."
    if "deepspeed" in lower:
        return f"{summary} Root cause: DeepSpeed launcher or engine initialization failure."
    if "lm-eval: error" in lower:
        return f"{summary} Root cause: lm_eval CLI argument parsing failure."
    return f"{summary} Root cause: see error summary."


def build_command(
    codec: str,
    task: str,
    master_port: int,
    hs_max_gen_len: int | None = None,
    lm_eval_output_path: Path | None = None,
    log_samples: bool = False,
) -> tuple[list[str], str]:
    py_path = (
        "/home/jt/inference_acrhitecture/third-party/DeepSpeed:"
        "/home/jt/inference_acrhitecture/third-party/transformers/src"
    )
    model_args_parts = [
        "pretrained=/home/share/models/Llama-2-7b-hf",
        "hs_offload=True",
        "hs_device=cpu",
        f"hs_codec_name={codec}",
        "short_context_exact_threshold=16",
        "dtype=float16",
        "ds_batch_size=1",
    ]
    if hs_max_gen_len is not None:
        model_args_parts.append(f"hs_max_gen_len={int(hs_max_gen_len)}")
    model_args = ",".join(model_args_parts)
    cmd = [
        "conda",
        "run",
        "-n",
        "deepspeed_v1",
        "env",
        f"PYTHONPATH={py_path}",
        "deepspeed",
        "--num_gpus",
        "1",
        "--master_port",
        str(master_port),
        "--no_local_rank",
        "--module",
        "lm_eval",
        "--model",
        "hs-offload-hf",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--device",
        "cuda:0",
        "--batch_size",
        "1",
    ]
    if log_samples:
        assert lm_eval_output_path is not None
        cmd.extend(
            [
                "--log_samples",
                "--output_path",
                str(lm_eval_output_path),
            ]
        )
    return cmd, shlex.join(cmd)


def locate_latest_json(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def locate_latest_artifact(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(f"**/{pattern}"))
    if not matches:
        return None
    return matches[-1]


def extract_first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return extract_first_text(value[0])
    return ""


def load_sample_outputs(sample_path: Path, field: str) -> dict[int, str]:
    samples: dict[int, str] = {}
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            doc_id = int(sample["doc_id"])
            samples[doc_id] = extract_first_text(sample.get(field, ""))
    return samples


def compare_sample_outputs(
    baseline_sample_path: Path,
    candidate_sample_path: Path,
    field: str,
) -> dict[str, str]:
    baseline = load_sample_outputs(baseline_sample_path, field=field)
    candidate = load_sample_outputs(candidate_sample_path, field=field)
    shared_doc_ids = sorted(set(baseline) & set(candidate))
    if not shared_doc_ids:
        return {
            "sample_compared_count": "0",
            "sample_same_count": "0",
            "sample_diff_count": "0",
            "sample_same_ratio": "",
            "sample_first_diff_doc_id": "",
        }

    same_count = 0
    first_diff_doc_id = ""
    for doc_id in shared_doc_ids:
        if baseline[doc_id] == candidate[doc_id]:
            same_count += 1
        elif not first_diff_doc_id:
            first_diff_doc_id = str(doc_id)

    compared_count = len(shared_doc_ids)
    diff_count = compared_count - same_count
    same_ratio = same_count / compared_count if compared_count else 0.0
    return {
        "sample_compared_count": str(compared_count),
        "sample_same_count": str(same_count),
        "sample_diff_count": str(diff_count),
        "sample_same_ratio": f"{same_ratio:.4f}",
        "sample_first_diff_doc_id": first_diff_doc_id,
    }


def run_codec(
    codec: str,
    task: str,
    output_dir: Path,
    timeout_sec: int,
    master_port: int,
    hs_max_gen_len: int | None = None,
    log_samples: bool = False,
    sample_field: str = "filtered_resps",
) -> dict[str, str]:
    lm_eval_output_path = output_dir / "lm_eval_outputs" / codec if log_samples else None
    cmd, cmd_str = build_command(
        codec=codec,
        task=task,
        master_port=master_port,
        hs_max_gen_len=hs_max_gen_len,
        lm_eval_output_path=lm_eval_output_path,
        log_samples=log_samples,
    )
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n[TIMEOUT]\n" + (exc.stderr or "")
        returncode = 124
        timed_out = True

    elapsed = time.monotonic() - start
    log_path = output_dir / f"{codec}.log"
    log_path.write_text(output, encoding="utf-8")

    status = "success" if returncode == 0 else "failed"
    if timed_out:
        status = "failed"

    results_path = (
        locate_latest_artifact(lm_eval_output_path, "results_*.json")
        if lm_eval_output_path is not None and lm_eval_output_path.exists()
        else None
    )
    sample_path = (
        locate_latest_artifact(lm_eval_output_path, f"samples_{task}_*.jsonl")
        if lm_eval_output_path is not None and lm_eval_output_path.exists()
        else None
    )

    result = {
        "codec": codec,
        "status": status,
        "task": task,
        "command": cmd_str,
        "elapsed_time_sec": f"{elapsed:.2f}",
        "notes": determine_notes(output, task, status),
        "returncode": str(returncode),
        "log_path": str(log_path),
        "results_path": str(results_path) if results_path is not None else "",
        "sample_path": str(sample_path) if sample_path is not None else "",
        "sample_field": sample_field if log_samples else "",
        "sample_compared_count": "",
        "sample_same_count": "",
        "sample_diff_count": "",
        "sample_same_ratio": "",
        "sample_first_diff_doc_id": "",
    }
    for metric in metrics_for_task(task):
        parsed_metric = ""
        if status == "success":
            if results_path is not None and results_path.exists():
                parsed_metric = parse_metric_from_results_json(
                    results_path=results_path,
                    task=task,
                    metric=metric,
                )
            if not parsed_metric:
                parsed_metric = parse_metric(output, task, metric)
        result[metric] = parsed_metric
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    baseline_sample_path: Path | None = None
    for index, codec in enumerate(args.codecs):
        master_port = 29500 + index
        result = run_codec(
            codec=codec,
            task=args.task,
            output_dir=output_dir,
            timeout_sec=args.timeout_sec,
            master_port=master_port,
            hs_max_gen_len=args.hs_max_gen_len,
            log_samples=args.log_samples,
            sample_field=args.sample_field,
        )
        sample_path = Path(result["sample_path"]) if result["sample_path"] else None
        if codec == "fp16" and sample_path is not None and sample_path.exists():
            baseline_sample_path = sample_path
        elif (
            baseline_sample_path is not None
            and sample_path is not None
            and sample_path.exists()
            and result["status"] == "success"
        ):
            result.update(
                compare_sample_outputs(
                    baseline_sample_path=baseline_sample_path,
                    candidate_sample_path=sample_path,
                    field=args.sample_field,
                )
            )
        results.append(result)
        print(
            json.dumps(
                {
                    "codec": result["codec"],
                    "status": result["status"],
                    "elapsed_time_sec": result["elapsed_time_sec"],
                    "metrics": {
                        metric: result.get(metric, "")
                        for metric in metrics_for_task(args.task)
                    },
                    "sample_diff_count": result["sample_diff_count"],
                    "sample_same_ratio": result["sample_same_ratio"],
                    "returncode": result["returncode"],
                    "log_path": result["log_path"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "codec",
                "status",
                "task",
                "command",
                "elapsed_time_sec",
                "acc",
                "acc_norm",
                "perplexity",
                "em",
                "f1",
                "exact_match",
                "notes",
                "returncode",
                "log_path",
                "results_path",
                "sample_path",
                "sample_field",
                "sample_compared_count",
                "sample_same_count",
                "sample_diff_count",
                "sample_same_ratio",
                "sample_first_diff_doc_id",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
