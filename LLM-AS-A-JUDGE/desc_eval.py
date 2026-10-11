import json
import sys
from pathlib import Path
from typing import Any, Dict

import ollama
from pydantic import BaseModel, Field

# Configuration
MODEL = "qwen3-vl:8b"
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "evaluation"

# Domain-specific scoring weights for descriptions
WEIGHTS = {
    "semantic_accuracy": 0.35,
    "hallucination_control": 0.35,
    "completeness": 0.15,
    "semantic_richness": 0.15,
}

# Temperature & seed for deterministic execution
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "seed": 42,
    "num_ctx": 8192
}


class DescriptionEvaluationResult(BaseModel):
    semantic_accuracy: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    hallucination_control: float = Field(ge=1.0, le=5.0)
    semantic_richness: float = Field(ge=1.0, le=5.0)
    reasoning: str


class TripletEvaluationResult(BaseModel):
    subject_validity: float = Field(ge=1.0, le=5.0)
    predicate_correctness: float = Field(ge=1.0, le=5.0)
    object_accuracy: float = Field(ge=1.0, le=5.0)
    triplet_hallucination_control: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    reasoning: str


def build_description_prompt(description: str) -> str:
    return f"""You are an expert AI evaluator for image description tasks in UI/OSINT domains.

Evaluate the generated description relative to the image source.

EVALUATION PRINCIPLES:
- Critical Entities (Persons, Organizations, Accounts, OCR text) carry high weight.
- Secondary elements (icons, borders) carry low weight.

SCALE CALIBRATION (1.0 to 5.0):
- Use the FULL score range dynamically.
- Do NOT compress scores toward the middle.
- If an output is 100% accurate, complete, and fully grounded with zero errors, award a full **5.0**.

Generated Description:
"{description}"

SCORING CRITERIA:
- semantic_accuracy: 1.0 = Severe factual errors; 5.0 = Fully accurate.
- completeness: 1.0 = Major omissions; 5.0 = Complete coverage of key entities.
- hallucination_control: 1.0 = Fabricated data; 5.0 = Zero hallucinations.
- semantic_richness: 1.0 = Extremely vague; 5.0 = Rich and precise.

Return ONLY a valid JSON object:
{{
  "semantic_accuracy": 1.0-5.0,
  "completeness": 1.0-5.0,
  "hallucination_control": 1.0-5.0,
  "semantic_richness": 1.0-5.0,
  "reasoning": "High-level summary of findings."
}}"""


def build_triplet_prompt(triplets_json: str) -> str:
    return f"""You are an expert AI evaluator for Knowledge Graph Triplet extraction.

Evaluate extracted triplets against visible UI evidence.

SCALE CALIBRATION (1.0 to 5.0):
- Use the FULL score range strictly based on objective correctness.
- Do NOT artificially depress flawless extractions. If all subjects, predicates, and values are 100% accurate and no facts are missing or fabricated, award a full **5.0**.
- Penalize ONLY when there is explicit visual evidence of errors, hallucinations, or missing entities.

Extracted Triplets:
{triplets_json}

SCORING CRITERIA:
- subject_validity: Validity of subjects (1.0-5.0).
- predicate_correctness: Precision of relations (1.0-5.0).
- object_accuracy: Accuracy of extracted values/URIs (1.0-5.0).
- triplet_hallucination_control: Absence of non-existent facts (1.0-5.0).
- completeness: Coverage of prominent visible relations in the UI (1.0-5.0).

Return ONLY a valid JSON object:
{{
  "subject_validity": 1.0-5.0,
  "predicate_correctness": 1.0-5.0,
  "object_accuracy": 1.0-5.0,
  "triplet_hallucination_control": 1.0-5.0,
  "completeness": 1.0-5.0,
  "reasoning": "High-level summary of triplet precision and completeness."
}}"""


def evaluate_description(image_path: Path, description: str) -> Dict[str, Any]:
    max_retries = 2
    content = ""

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": build_description_prompt(description),
                    "images": [str(image_path)],
                }],
                options=OLLAMA_OPTIONS,
                format="json",
            )

            if response.message and response.message.content:
                content = response.message.content.strip()
                if content:
                    break
        except Exception as e:
            print(f"  [RETRY Desc] Attempt {attempt + 1}/{max_retries}: {e}")

    if not content:
        raise RuntimeError("Empty response from Ollama.")

    result = DescriptionEvaluationResult.model_validate(json.loads(content))

    overall = (
        (result.semantic_accuracy * WEIGHTS["semantic_accuracy"])
        + (result.hallucination_control * WEIGHTS["hallucination_control"])
        + (result.completeness * WEIGHTS["completeness"])
        + (result.semantic_richness * WEIGHTS["semantic_richness"])
    )

    return {
        "semantic_accuracy": round(result.semantic_accuracy, 2),
        "completeness": round(result.completeness, 2),
        "hallucination_control": round(result.hallucination_control, 2),
        "semantic_richness": round(result.semantic_richness, 2),
        "overall_score": round(overall, 2),
        "reasoning": result.reasoning,
    }


def evaluate_triplets(image_path: Path, triplets_data: Any) -> Dict[str, Any]:
    max_retries = 2
    content = ""
    triplets_str = (
        json.dumps(triplets_data, indent=2)
        if not isinstance(triplets_data, str)
        else triplets_data
    )

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": build_triplet_prompt(triplets_str),
                    "images": [str(image_path)],
                }],
                options=OLLAMA_OPTIONS,
                format="json",
            )

            if response.message and response.message.content:
                content = response.message.content.strip()
                if content:
                    break
        except Exception as e:
            print(f"  [RETRY Triplets] Attempt {attempt + 1}/{max_retries}: {e}")

    if not content:
        raise RuntimeError("Empty response from Ollama.")

    result = TripletEvaluationResult.model_validate(json.loads(content))

    scores = [
        result.subject_validity,
        result.predicate_correctness,
        result.object_accuracy,
        result.triplet_hallucination_control,
        result.completeness,
    ]
    overall = sum(scores) / len(scores)

    return {
        "subject_validity": round(result.subject_validity, 2),
        "predicate_correctness": round(result.predicate_correctness, 2),
        "object_accuracy": round(result.object_accuracy, 2),
        "triplet_hallucination_control": round(result.triplet_hallucination_control, 2),
        "completeness": round(result.completeness, 2),
        "overall_score": round(overall, 2),
        "reasoning": result.reasoning,
    }


def generate_markdown_report(
    metadata: Dict[str, Any],
    summary: Dict[str, Any],
    detailed_results: Dict[str, Any],
) -> str:
    md = ["# LLM-as-a-Judge Evaluation Report\n"]

    md.append("## Metadata")
    md.append(f"- **Evaluator:** {metadata['evaluator']}")
    md.append(f"- **Model:** {metadata['model']}")
    md.append(f"- **Total Samples:** {metadata['total_samples']}\n")

    md.append("## Benchmark Summary Averages\n")
    if "description_averages" in summary:
        md.append("### Description Metrics")
        md.append("| Metric | Average Score (/5.0) |")
        md.append("| :--- | :---: |")
        for k, v in summary["description_averages"].items():
            md.append(f"| `{k}` | **{v}** |")
        md.append("")

    if "triplet_averages" in summary:
        md.append("### Knowledge Graph Metrics")
        md.append("| Metric | Average Score (/5.0) |")
        md.append("| :--- | :---: |")
        for k, v in summary["triplet_averages"].items():
            md.append(f"| `{k}` | **{v}** |")
        md.append("")

    md.append("---\n## Detailed Evaluations\n")

    for stem, sample in detailed_results.items():
        md.append(f"### Sample: `{sample['image_filename']}`\n")

        if "description_evaluation" in sample:
            d_eval = sample["description_evaluation"]
            md.append("#### Description Evaluation")
            md.append("| Metric | Score (/5.0) |")
            md.append("| :--- | :---: |")
            md.append(f"| Semantic Accuracy | {d_eval['semantic_accuracy']} |")
            md.append(f"| Hallucination Control | {d_eval['hallucination_control']} |")
            md.append(f"| Completeness | {d_eval['completeness']} |")
            md.append(f"| Semantic Richness | {d_eval['semantic_richness']} |")
            md.append(f"| **Overall Score** | **{d_eval['overall_score']}** |")
            md.append(f"\n> **Rationale:** {d_eval['reasoning']}\n")

        if "triplet_evaluation" in sample:
            t_eval = sample["triplet_evaluation"]
            md.append("#### Knowledge Graph Evaluation")
            md.append("| Metric | Score (/5.0) |")
            md.append("| :--- | :---: |")
            md.append(f"| Subject Validity | {t_eval['subject_validity']} |")
            md.append(f"| Predicate Correctness | {t_eval['predicate_correctness']} |")
            md.append(f"| Object Accuracy | {t_eval['object_accuracy']} |")
            md.append(f"| Triplet Hallucination Control | {t_eval['triplet_hallucination_control']} |")
            md.append(f"| Completeness | {t_eval['completeness']} |")
            md.append(f"| **Overall Score** | **{t_eval['overall_score']}** |")
            md.append(f"\n> **Rationale:** {t_eval['reasoning']}\n")

        md.append("---\n")

    return "\n".join(md)


def main():
    print(f"Model: {MODEL}")
    try:
        ollama.show(MODEL)
    except Exception as e:
        print(f"[ERROR] Model unavailable: {e}")
        sys.exit(1)

    image_files = sorted(
        list(IMAGES_DIR.glob("*.png"))
        + list(IMAGES_DIR.glob("*.jpg"))
        + list(IMAGES_DIR.glob("*.jpeg"))
        + list(IMAGES_DIR.glob("*.webp"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {IMAGES_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    desc_metrics_sum = {
        "semantic_accuracy": 0.0,
        "completeness": 0.0,
        "hallucination_control": 0.0,
        "semantic_richness": 0.0,
        "overall_score": 0.0,
    }
    triplet_metrics_sum = {
        "subject_validity": 0.0,
        "predicate_correctness": 0.0,
        "object_accuracy": 0.0,
        "triplet_hallucination_control": 0.0,
        "completeness": 0.0,
        "overall_score": 0.0,
    }

    successful_desc = 0
    successful_triplets = 0

    for i, image_path in enumerate(image_files, 1):
        stem = image_path.stem
        desc_path = RESULTS_DIR / f"{stem}_desc.txt"
        triplet_path = RESULTS_DIR / f"{stem}_graph.json"

        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
        sample_res = {"image_filename": image_path.name}

        if desc_path.exists():
            description = desc_path.read_text(encoding="utf-8").strip()
            try:
                desc_eval = evaluate_description(image_path, description)
                sample_res["description_evaluation"] = desc_eval
                for k in desc_metrics_sum:
                    desc_metrics_sum[k] += desc_eval[k]
                successful_desc += 1
                print(f"  [Description] Score: {desc_eval['overall_score']}/5.0")
            except Exception as e:
                print(f"  [ERROR Desc] {e}")

        if triplet_path.exists():
            try:
                triplets_data = json.loads(triplet_path.read_text(encoding="utf-8"))
                triplet_eval = evaluate_triplets(image_path, triplets_data)
                sample_res["triplet_evaluation"] = triplet_eval
                for k in triplet_metrics_sum:
                    triplet_metrics_sum[k] += triplet_eval[k]
                successful_triplets += 1
                print(f"  [Triplets] Score: {triplet_eval['overall_score']}/5.0")
            except Exception as e:
                print(f"  [ERROR Triplets] {e}")

        results[stem] = sample_res

    summary = {}
    if successful_desc > 0:
        summary["description_averages"] = {
            k: round(v / successful_desc, 2) for k, v in desc_metrics_sum.items()
        }
    if successful_triplets > 0:
        summary["triplet_averages"] = {
            k: round(v / successful_triplets, 2) for k, v in triplet_metrics_sum.items()
        }

    metadata = {
        "evaluator": "Ollama",
        "model": MODEL,
        "total_samples": len(image_files),
        "successful_descriptions": successful_desc,
        "successful_triplets": successful_triplets,
    }

    markdown_report = generate_markdown_report(metadata, summary, results)
    output_path = OUTPUT_DIR / "llm_judge_weighted_report.md"
    output_path.write_text(markdown_report, encoding="utf-8")

    print("\n" + "=" * 50)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()