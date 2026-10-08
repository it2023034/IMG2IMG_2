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

# Domain-specific scoring weights for description overall score calculation
WEIGHTS = {
    "semantic_accuracy": 0.35,
    "hallucination_control": 0.35,
    "completeness": 0.15,
    "semantic_richness": 0.15,
}


# Structured evaluator output for Text Descriptions (Supports floats: 1.0 - 5.0)
class DescriptionEvaluationResult(BaseModel):
    semantic_accuracy: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    hallucination_control: float = Field(ge=1.0, le=5.0)
    semantic_richness: float = Field(ge=1.0, le=5.0)
    reasoning: str


# Structured evaluator output for Knowledge Graph Triplets (Supports floats: 1.0 - 5.0)
class TripletEvaluationResult(BaseModel):
    subject_validity: float = Field(ge=1.0, le=5.0)
    predicate_correctness: float = Field(ge=1.0, le=5.0)
    object_accuracy: float = Field(ge=1.0, le=5.0)
    triplet_hallucination_control: float = Field(ge=1.0, le=5.0)
    reasoning: str


def build_description_prompt(description: str) -> str:
    return f"""You are an expert, impartial AI evaluator acting as a strict automated judge for multimodal image description tasks in UI/OSINT domains.

Evaluate the provided generated description objectively, measuring its semantic accuracy and usefulness relative to the visual source.

EVALUATION PRINCIPLES & DOMAIN WEIGHTING:
1. High-Value Critical Entities (WEIGHT: HIGH):
   - Persons, Organizations, Account Pages, User Identifiers, and Explicit OCR Text Strings are CRITICAL.
   - Any hallucination, misspelling, or severe omission regarding these elements MUST severely penalize 'semantic_accuracy' and 'hallucination_control' (max 1.0-2.5 score).

2. Secondary Visual Elements (WEIGHT: LOW):
   - Visual symbols, decorative icons, UI borders, and general layout spacing are SECONDARY.
   - Minor inaccuracies or omissions here should ONLY marginally impact scores if all Critical Entities are 100% accurate.

3. Visual Grounding & Precision: Claims must be grounded in information visibly present in the image. Do not reward verbosity.

Generated Description to Evaluate:
"{description}"

SCORING CRITERIA (1.0 to 5.0 scale, decimals like 3.5 or 4.2 are allowed):
- semantic_accuracy: 1.0 = Severe factual errors in critical entities/text; 5.0 = Fully accurate and faithful.
- completeness: 1.0 = Omits critical OSINT/UI entities; 5.0 = Captures essentially all important context.
- hallucination_control: 1.0 = Fabricated names, organisations, or UI data; 5.0 = Fully grounded without hallucinations.
- semantic_richness: 1.0 = Extremely vague/generic; 5.0 = Highly informative, precise, and semantically rich.

Return ONLY a valid JSON object matching this exact structure:
{{
  "semantic_accuracy": 1.0-5.0,
  "completeness": 1.0-5.0,
  "hallucination_control": 1.0-5.0,
  "semantic_richness": 1.0-5.0,
  "reasoning": "A concise justification highlighting matches/errors in critical entities vs secondary symbols."
}}"""


def build_triplet_prompt(triplets_json: str) -> str:
    return f"""You are an expert, impartial AI evaluator acting as a strict automated judge for Knowledge Graph Triplet extraction from UI/OSINT images.

Evaluate the extracted Knowledge Graph Triplets (Subject, Predicate, Object) relative to the visible UI elements and text in the image.

EVALUATION PRINCIPLES:
1. High-Value Entity Priority: Triplets involving 'person', 'organisation', or 'account' require 100% precision.
2. Predicate Grounding: Predicates (e.g., 'contains', 'has_text', 'located_in') must accurately reflect the visual topology or semantics.
3. Object Precision: Text values or associated entities must strictly match the OCR/UI evidence.

Extracted Triplets to Evaluate:
{triplets_json}

SCORING CRITERIA (1.0 to 5.0 scale, decimals like 3.5 or 4.2 are allowed):
- subject_validity: 1.0 = Subjects do not exist or misidentify UI elements; 5.0 = Subjects are correctly identified.
- predicate_correctness: 1.0 = Predicates are semantically invalid; 5.0 = Predicates accurately describe relationships.
- object_accuracy: 1.0 = Values/objects are incorrect or fabricated; 5.0 = Values/objects match visual evidence exactly.
- triplet_hallucination_control: 1.0 = Extracted triplets are entirely fabricated; 5.0 = Every triplet is grounded in the image.

Return ONLY a valid JSON object matching this exact structure:
{{
  "subject_validity": 1.0-5.0,
  "predicate_correctness": 1.0-5.0,
  "object_accuracy": 1.0-5.0,
  "triplet_hallucination_control": 1.0-5.0,
  "reasoning": "A concise justification evaluating triplet precision, entity types, and visual grounding."
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
                options={"temperature": 0, "num_ctx": 8192},
                format="json",
            )

            if response.message and response.message.content:
                content = response.message.content.strip()
                if content:
                    break
        except Exception as e:
            print(f"  [RETRY Desc] Attempt {attempt + 1}/{max_retries}: {e}")

    if not content:
        raise RuntimeError("Empty response from Ollama after retries.")

    result = DescriptionEvaluationResult.model_validate(json.loads(content))

    # Calculate weighted overall score
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


def evaluate_triplets(
    image_path: Path, triplets_data: Any
) -> Dict[str, Any]:
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
                options={"temperature": 0, "num_ctx": 8192},
                format="json",
            )

            if response.message and response.message.content:
                content = response.message.content.strip()
                if content:
                    break
        except Exception as e:
            print(
                f"  [RETRY Triplets] Attempt {attempt + 1}/{max_retries}: {e}"
            )

    if not content:
        raise RuntimeError(
            "Empty response from Ollama during triplet evaluation."
        )

    result = TripletEvaluationResult.model_validate(json.loads(content))

    scores = [
        result.subject_validity,
        result.predicate_correctness,
        result.object_accuracy,
        result.triplet_hallucination_control,
    ]
    overall = sum(scores) / len(scores)

    return {
        "subject_validity": round(result.subject_validity, 2),
        "predicate_correctness": round(result.predicate_correctness, 2),
        "object_accuracy": round(result.object_accuracy, 2),
        "triplet_hallucination_control": round(result.triplet_hallucination_control, 2),
        "overall_score": round(overall, 2),
        "reasoning": result.reasoning,
    }


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
        "overall_score": 0.0,
    }

    successful_desc = 0
    successful_triplets = 0

    for i, image_path in enumerate(image_files, 1):
        stem = image_path.stem
        desc_path = RESULTS_DIR / f"{stem}_desc.txt"
        triplet_path = RESULTS_DIR / f"{stem}_triplets.json"

        print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
        sample_res = {"image_filename": image_path.name}

        # 1. Evaluate Description if present
        if desc_path.exists():
            description = desc_path.read_text(encoding="utf-8").strip()
            try:
                desc_eval = evaluate_description(image_path, description)
                sample_res["description_evaluation"] = desc_eval
                for k in desc_metrics_sum:
                    desc_metrics_sum[k] += desc_eval[k]
                successful_desc += 1
                print(
                    f"  [Description] Weighted Score: {desc_eval['overall_score']}/5.0 | Reasoning: {desc_eval['reasoning']}"
                )
            except Exception as e:
                print(f"  [ERROR Desc] {e}")

        # 2. Evaluate Triplets if present
        if triplet_path.exists():
            try:
                triplets_data = json.loads(
                    triplet_path.read_text(encoding="utf-8")
                )
                triplet_eval = evaluate_triplets(image_path, triplets_data)
                sample_res["triplet_evaluation"] = triplet_eval
                for k in triplet_metrics_sum:
                    triplet_metrics_sum[k] += triplet_eval[k]
                successful_triplets += 1
                print(
                    f"  [Triplets] Score: {triplet_eval['overall_score']}/5.0 | Reasoning: {triplet_eval['reasoning']}"
                )
            except Exception as e:
                print(f"  [ERROR Triplets] {e}")

        results[stem] = sample_res

    # Aggregations
    summary = {}
    if successful_desc > 0:
        summary["description_averages"] = {
            k: round(v / successful_desc, 2)
            for k, v in desc_metrics_sum.items()
        }
    if successful_triplets > 0:
        summary["triplet_averages"] = {
            k: round(v / successful_triplets, 2)
            for k, v in triplet_metrics_sum.items()
        }

    report = {
        "metadata": {
            "evaluator": "Ollama",
            "model": MODEL,
            "total_samples": len(image_files),
            "successful_descriptions": successful_desc,
            "successful_triplets": successful_triplets,
        },
        "summary_averages": summary,
        "detailed_evaluations": results,
    }

    output_path = OUTPUT_DIR / "llm_judge_weighted_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 50)
    if "description_averages" in summary:
        print("DESCRIPTION METRICS (Weighted):")
        for k, v in summary["description_averages"].items():
            print(f"  {k:30s}: {v}/5.0")
    if "triplet_averages" in summary:
        print("\nKNOWLEDGE GRAPH TRIPLET METRICS:")
        for k, v in summary["triplet_averages"].items():
            print(f"  {k:30s}: {v}/5.0")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()