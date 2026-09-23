"""Measure classification price on a few trace lengths and scale it to the test set.

The original run did not store token or dollar usage. OpenRouter activity
requires a management key, so this script records cost on six length
quantiles and fits a line of price against trace length.
"""

import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

import run_eval

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "test_qt.csv"
SAMPLES_PATH = ROOT / "results" / "cost_samples.csv"
SUMMARY_PATH = ROOT / "results" / "summary.csv"
JEV_USD_PER_MILLION_INPUT = 0.042
QUANTILES = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def post(url, headers, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def call_jev(env, query, trace):
    body = post(
        "https://api.typesafe.ai/v1/systemone",
        {
            "Authorization": "Bearer " + env["JEV_API_KEY"],
            "Content-Type": "application/json",
        },
        {
            "model": run_eval.MODEL_IDS["Jev"],
            "state": {"query": query, "reasoning_trace": trace},
            "questions": {
                "safety_level": {
                    "type": "choice",
                    "instructions": run_eval.INSTRUCTIONS,
                    "criteria": run_eval.CRITERIA,
                }
            },
        },
    )
    tokens = body["usage"]["input_tokens"]
    return {
        "input_tokens": tokens,
        "output_tokens": body["usage"].get("output_tokens", 0),
        "cost_usd": tokens * JEV_USD_PER_MILLION_INPUT / 1_000_000,
    }


def call_llm(env, classifier, query, trace):
    body = post(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": "Bearer " + env["OPENROUTHER_API_KEY"],
            "Content-Type": "application/json",
        },
        {
            "model": run_eval.MODEL_IDS[classifier],
            "temperature": 0,
            "max_tokens": 40,
            "reasoning": {"effort": "none"},
            "messages": [
                {"role": "system", "content": run_eval.LLM_SYSTEM},
                {"role": "user", "content": "Query:\n" + query + "\n\nReasoning trace:\n" + trace},
            ],
        },
    )
    usage = body.get("usage") or {}
    if "cost" not in usage:
        raise ValueError("missing cost")
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "cost_usd": usage["cost"],
    }


def fit_line(xs, ys):
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    variance = sum((x - mean_x) ** 2 for x in xs)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    return intercept, slope


def choose_rows(rows):
    ordered = sorted(rows, key=lambda row: len(row["query"]) + len(row["model_thinking"]))
    picked = []
    seen = set()
    for quantile in QUANTILES:
        index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
        while index in seen and index + 1 < len(ordered):
            index += 1
        seen.add(index)
        picked.append(ordered[index])
    return picked


def main():
    env = load_env()
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    selected = choose_rows(dataset)
    samples = []
    for row in selected:
        chars = len(row["query"]) + len(row["model_thinking"])
        for classifier in run_eval.CLASSIFIERS:
            try:
                if classifier == "Jev":
                    measured = call_jev(env, row["query"], row["model_thinking"])
                else:
                    measured = call_llm(env, classifier, row["query"], row["model_thinking"])
                samples.append(
                    {
                        "sample_id": row["index"],
                        "classifier": classifier,
                        "trace_chars": chars,
                        "input_tokens": measured["input_tokens"],
                        "output_tokens": measured["output_tokens"],
                        "cost_usd": round(measured["cost_usd"], 8),
                    }
                )
                print(
                    classifier,
                    "chars",
                    chars,
                    "tokens",
                    measured["input_tokens"],
                    "usd",
                    round(measured["cost_usd"], 6),
                    flush=True,
                )
            except Exception as error:
                detail = ""
                if isinstance(error, urllib.error.HTTPError):
                    detail = error.read().decode(errors="replace")[:160].replace("\n", " ")
                print("fail", classifier, "chars", chars, type(error).__name__, detail, flush=True)

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "classifier", "trace_chars", "input_tokens", "output_tokens", "cost_usd"],
        )
        writer.writeheader()
        writer.writerows(samples)

    lengths = [len(row["query"]) + len(row["model_thinking"]) for row in dataset]
    summary_rows = []
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] not in {"mean_cost_usd", "total_cost_usd"}:
                summary_rows.append(row)

    print("FIT")
    for classifier in run_eval.CLASSIFIERS:
        points = [row for row in samples if row["classifier"] == classifier]
        if len(points) < 2:
            print(classifier, "not enough points", len(points))
            continue
        intercept, slope = fit_line(
            [int(row["trace_chars"]) for row in points],
            [float(row["cost_usd"]) for row in points],
        )
        predicted = [max(0.0, intercept + slope * length) for length in lengths]
        mean_cost = sum(predicted) / len(predicted)
        total_cost = sum(predicted)
        print(
            classifier,
            "n",
            len(points),
            "mean_usd",
            round(mean_cost, 6),
            "total_usd",
            round(total_cost, 4),
            "per_1000",
            round(mean_cost * 1000, 4),
        )
        summary_rows.append(
            {
                "classifier": classifier,
                "metric": "mean_cost_usd",
                "group": "overall",
                "group_value": "per_call",
                "value": round(mean_cost, 8),
                "unit": "usd",
                "n": len(predicted),
            }
        )
        summary_rows.append(
            {
                "classifier": classifier,
                "metric": "total_cost_usd",
                "group": "overall",
                "group_value": "test_set",
                "value": round(total_cost, 6),
                "unit": "usd",
                "n": len(predicted),
            }
        )

    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["classifier", "metric", "group", "group_value", "value", "unit", "n"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
