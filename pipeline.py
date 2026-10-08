import gc
import json
import os
import re
import model
import numpy as np
import prompts
import torch
import utils

# Define directory structures and schema paths
RESULTS_DIR = "results"
CF_RESULTS_DIR = "counterfactual_results"
IMAGES_DIR = "images"
CF_IMAGES_DIR = "counterfactual"
SCHEMA_PATH = "visual_metapath.txt"
EXAMPLES_PATH = "triples_example.txt"


def extract_original_triples():
    """Extracts descriptions, Named Entities (NER), and relations from original images,

    saving the result as a Knowledge Graph JSON.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Load ontology schema if available
    schema_content = ""
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_content = f.read()
    elif os.path.exists("prompts/schema.txt"):
        with open("prompts/schema.txt", "r", encoding="utf-8") as f:
            schema_content = f.read()

    # Load relation extraction examples
    examples_content = ""
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            examples_content = f.read()

    # Collect and sort supported images
    image_list = sorted(
        [
            f
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    )

    for img_name in image_list:
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"\n==========================================")
        print(f"Processing Image: {img_name}")
        print(f"==========================================")

        # 1. Generate text description via VLM
        description = model.generate_description(
            img_path, prompts.get_description_prompt()
        )
        utils.save_description_txt(description, img_name, RESULTS_DIR)

        # 2. Extract and clean Named Entities (NER)
        print(f"Extracting entities (NER) for {img_name}...")
        raw_ner_str = model.extract_entities(
            description, prompts.get_ner_prompt
        )

        raw_ner_triples = utils.parse_triples(raw_ner_str)
        cleaned_ner = utils.clean_and_deduplicate(raw_ner_triples)

        # Filter entities based on allowed ontology classes
        if hasattr(utils, "apply_dynamic_ner_filtering"):
            filtered_ner = utils.apply_dynamic_ner_filtering(
                img_name, cleaned_ner
            )
        else:
            allowed_classes = utils.extract_allowed_classes_from_schema(
                schema_content
            )
            filtered_ner = utils.filter_ner_by_allowed_classes(
                cleaned_ner, allowed_classes
            )

        ner_list, id_map = utils.deduplicate_entities_by_label(filtered_ner)

        # 3. Extract knowledge relations between entities
        print(f"Extracting relations for {img_name}...")
        filtered_ner_str = "\n".join(
            [
                f"{e['subject']} | {e['predicate']} | {e['object']}"
                for e in ner_list
            ]
        )

        raw_relations_str = model.extract_triples(
            description,
            schema_content,
            examples_content,
            filtered_ner_str,
            lambda d, s, e, n: prompts.get_relation_extraction_prompt(
                d, s, e, n
            ),
        )

        # Validate and build final graph connections
        final_entities, valid_relations = utils.process_pipeline_relations(
            raw_relations_str,
            ner_list,
            schema_content,
            id_map,
            img_name=img_name,
        )

        # 4. Save the compiled Knowledge Graph
        base_name = os.path.splitext(img_name)[0]
        graph_entry = {
            "image": img_name,
            "description": description,
            "entities": final_entities,
            "relations": valid_relations,
        }

        graph_entry = utils.post_process_graph(graph_entry)
        utils.save_json([graph_entry], f"{base_name}_graph.json", RESULTS_DIR)
        print(f"[SUCCESS] Saved original graph JSON for {img_name}")


def run_image_generation():
    """Selects a target entity, generates counterfactual text under geometric constraints,

    renders a modified image via FLUX, and generates visual audit highlights.
    """
    if not os.path.exists(CF_IMAGES_DIR):
        os.makedirs(CF_IMAGES_DIR)

    image_list = sorted(
        [
            f
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    )
    if not image_list:
        print("No images found in images/ directory.")
        return

    img_name = image_list[0]
    base_name = os.path.splitext(img_name)[0]
    json_path = os.path.join(RESULTS_DIR, f"{base_name}_graph.json")

    if not os.path.exists(json_path):
        print(
            f"Original graph JSON not found for {img_name}. Run option [1] first."
        )
        return

    with open(json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)[0]

    description = graph_data.get("description", "")
    entities = graph_data.get("entities", [])

    # Step A: Select target entity or visual attribute to modify
    cf_select_prompt = f"""
    Analyze the following image description and extracted entities.
    Select ONE specific target entity or visual attribute string (e.g. Person Name, Location, or Role) to modify.

    Description: "{description}"
    Extracted Entities: {json.dumps(entities)}

    Return ONLY a JSON object with keys "change_from", "target_entity", and "target_predicate":
    {{
        "change_from": "<exact string to modify, e.g. Nicolas Sanders>",
        "target_entity": "<subject ID, e.g. Person_1>",
        "target_predicate": "<predicate, e.g. label or rdfs:label>"
    }}
    """

    print("[INFO] Selecting target entity via LLM...")
    llm_response = model.generate_description([], cf_select_prompt)

    clean_change_from = ""
    target_entity = None
    target_predicate = "label"

    # Parse LLM response with graph label fallback
    try:
        json_clean = re.sub(r"```json\s*|\s*```", "", llm_response).strip()
        cf_target_data = json.loads(json_clean)
        clean_change_from = (
            cf_target_data.get("change_from", "").replace('"', "").strip()
        )
        target_entity = cf_target_data.get("target_entity", None)
        target_predicate = cf_target_data.get("target_predicate", "label")
    except Exception as e:
        print(
            f"[FALLBACK] Parsing failed ({e}). Extracting from graph labels..."
        )
        for ent in entities:
            if str(ent.get("predicate", "")).lower() in ["label", "rdfs:label"]:
                target_entity = ent.get("subject")
                target_predicate = ent.get("predicate")
                clean_change_from = (
                    str(ent.get("object", "")).replace('"', "").strip()
                )
                break

    if not clean_change_from:
        print("Could not find a valid target for modification.")
        return

    # Isolate target entity string
    matched_entity = None
    for ent in entities:
        ent_obj = str(ent.get("object", "")).strip()
        if ent_obj and ent_obj.lower() in clean_change_from.lower():
            if not ent_obj.startswith("http"):
                matched_entity = ent_obj
                break
            else:
                matched_entity = ent_obj.split("/")[-1].replace("_", " ")
                break

    if matched_entity:
        print(
            f"[ENTITY ISOLATED] Cleaned target string from '{clean_change_from}' -> '{matched_entity}'"
        )
        clean_change_from = matched_entity

    clean_change_from = re.sub(
        r"^(currently in|from|works at|lives in)\s+",
        "",
        clean_change_from,
        flags=re.IGNORECASE,
    ).strip()

    # Step B: Generate counterfactual value respecting character length and initials
    print(
        f"[INFO] Generating counterfactual using Strict Geometric Rules for: '{clean_change_from}'..."
    )

    target_len = len(clean_change_from)
    allowed_tolerance = 1  # Strict +/-1 character length tolerance
    min_len = max(1, target_len - allowed_tolerance)
    max_len = target_len + allowed_tolerance

    expected_words = [w for w in clean_change_from.split() if w]
    expected_initials = [w[0].upper() for w in expected_words]

    change_to_label = ""
    candidate = ""

    # Iterative retry loop for string constraint enforcement
    for attempt in range(5):
        strict_prompt = prompts.get_geometric_counterfactual_prompt(
            clean_change_from
        )

        if attempt > 0:
            strict_prompt += f"\n\nCRITICAL RETRY #{attempt+1}: Previous output '{candidate}' failed. Ensure real-world valid spelling AND length strictly between {min_len} and {max_len} chars."

        candidate = model.generate_description([], strict_prompt).strip(
            " \"'\n`"
        )
        candidate = re.sub(r"```.*?\n|\n```", "", candidate).strip()

        candidate_words = candidate.split()
        candidate_initials = [w[0].upper() for w in candidate_words if w]

        has_correct_initials = (
            len(candidate_initials) == len(expected_initials)
            and candidate_initials == expected_initials
        )
        has_correct_length = min_len <= len(candidate) <= max_len

        if has_correct_initials and has_correct_length:
            change_to_label = candidate
            break

        print(
            f"[RETRY {attempt+1}] LLM generated '{candidate}' ({len(candidate)} chars, Initials: {candidate_initials}). Required: {min_len}-{max_len} chars, Initials: {expected_initials}."
        )

    if not change_to_label:
        change_to_label = candidate

    print(f"\n[TARGET SELECTED & CONSTRAINED]")
    print(f"Target Entity : {target_entity}")
    print(f"Change From   : '{clean_change_from}' ({target_len} chars)")
    print(f"Change To     : '{change_to_label}' ({len(change_to_label)} chars)\n")

    # Persist temporary context for downstream graph updates
    temp_edit = {
        "original_image": img_name,
        "target_entity": target_entity,
        "target_predicate": target_predicate,
        "change_from": clean_change_from,
        "change_to": change_to_label,
    }
    with open("temporary_edit.json", "w", encoding="utf-8") as f:
        json.dump(temp_edit, f, indent=4, ensure_ascii=False)

    # Free VRAM from Ollama prior to FLUX execution
    print("Unloading Ollama VRAM to prepare for FLUX...")
    os.system(
        'curl -s http://localhost:11434/api/generate -d \'{"model": "gemma3:12b", "keep_alive": 0}\' > /dev/null 2>&1'
    )

    # Render counterfactual image using FLUX
    print("Loading FLUX Pipeline for Counterfactual Generation...")
    from pipeline_cf import CreateCounterfactualImage

    cf_generator = CreateCounterfactualImage()

    img_path = os.path.join(IMAGES_DIR, img_name)
    cf_image = cf_generator.generate(
        clean_change_from, change_to_label, img_path
    )

    cf_img_name = f"{base_name}_cf.png"
    cf_output_path = os.path.join(CF_IMAGES_DIR, cf_img_name)
    cf_image.save(cf_output_path)
    print(
        f"[SUCCESS] Counterfactual image saved inside '{cf_output_path}'!"
    )

    # --- VISUAL AUDIT: HIGHLIGHT BOTH ORIGINAL AND COUNTERFACTUAL IMAGES ---
    try:
        from highlight import CounterfactualHighlighter

        highlighter = CounterfactualHighlighter()

        # 1. Highlight on ORIGINAL image (targeting clean_change_from)
        orig_highlight_path = os.path.join(
            CF_IMAGES_DIR, f"{base_name}_orig_highlight.png"
        )
        highlighter.generate_highlight(
            img_path=img_path,
            target_text=clean_change_from,
            output_path=orig_highlight_path,
        )

        # 2. Highlight on COUNTERFACTUAL image (targeting change_to_label)
        cf_highlight_path = os.path.join(
            CF_IMAGES_DIR, f"{base_name}_cf_highlight.png"
        )
        highlighter.generate_highlight(
            img_path=cf_output_path,
            target_text=change_to_label,
            output_path=cf_highlight_path,
        )
    except Exception as e:
        print(f"[HIGHLIGHT ERROR] Failed to generate visual audit overlay: {e}")


def generate_counterfactual_triples():
    """Updates the Knowledge Graph JSON structure to align with the generated counterfactual image."""
    if not os.path.exists("temporary_edit.json"):
        print(
            "No active counterfactual generation context found. Run option [2] first."
        )
        return

    with open("temporary_edit.json", "r", encoding="utf-8") as f:
        temp_edit = json.load(f)

    img_name = temp_edit["original_image"]
    base_name = os.path.splitext(img_name)[0]

    with open(
        os.path.join(RESULTS_DIR, f"{base_name}_graph.json"),
        "r",
        encoding="utf-8",
    ) as f:
        original_graph_data = json.load(f)[0]

    detected_change = f"The original asset '{temp_edit['change_from']}' was replaced by '{temp_edit['change_to']}'."

    cf_entities = [dict(d) for d in original_graph_data["entities"]]
    cf_relations = [dict(d) for d in original_graph_data["relations"]]

    target_id = temp_edit["target_entity"]
    change_from = temp_edit["change_from"].lower()
    new_value = temp_edit["change_to"]

    clean_new_value = re.sub(
        r"^(currently in|from)\s+", "", new_value, flags=re.IGNORECASE
    ).strip()

    updated = False

    # 1. Update location DBpedia URIs directly inside relations (originates_from / associated_with_country)
    for rel in cf_relations:
        pred_lower = str(rel.get("predicate", "")).lower()
        if pred_lower in ["originates_from", "associated_with_country", "located_in"]:
            obj_val = str(rel.get("object", "")).lower()
            if any(
                loc in change_from
                for loc in obj_val.split("/")[-1].lower().split("_")
            ):
                formatted_uri = f"http://dbpedia.org/resource/{clean_new_value.replace(' ', '_')}"
                rel["object"] = formatted_uri
                updated = True
                print(
                    f"[LOCATION RELATION EDIT APPLIED] Updated relation {rel['subject']} -> {rel['predicate']} -> '{formatted_uri}'"
                )
                break

    # 2. Update literal objects in relations (e.g. profit, role, url, phone)
    if not updated:
        for rel in cf_relations:
            if target_id and rel["subject"] == target_id:
                rel_obj = str(rel["object"]).lower()
                if rel_obj in change_from or change_from in rel_obj:
                    rel["object"] = clean_new_value
                    updated = True
                    print(
                        f"[RELATION EDIT APPLIED] Updated relation {rel['subject']} -> {rel['predicate']} -> '{clean_new_value}'"
                    )
                    break

    # 3. Update entity label if edit applies to a Person/Organisation name
    if not updated:
        for ent in cf_entities:
            if (
                target_id
                and ent["subject"] == target_id
                and ent["predicate"] in ["label", "rdfs:label"]
            ):
                ent["object"] = clean_new_value
                updated = True
                print(
                    f"[ENTITY EDIT APPLIED] Updated entity {target_id} label to '{clean_new_value}'"
                )
                break

    # Fallback if no exact rule matched
    if not updated:
        print(
            "[WARNING] Could not automatically map the edit to the graph. Applying fallback to target entity label."
        )
        for ent in cf_entities:
            if target_id and ent["subject"] == target_id:
                ent["object"] = clean_new_value
                break

    # Compile modified Knowledge Graph output
    formatted_output = [
        {
            "counterfactual_image": f"{base_name}_cf.png",
            "detected_change": detected_change,
            "entities": cf_entities,
            "relations": cf_relations,
        }
    ]

    utils.save_json(
        formatted_output,
        f"{base_name}_counterfactual_results.json",
        CF_RESULTS_DIR,
    )
    print(
        f"[SUCCESS] Counterfactual JSON saved inside '{CF_RESULTS_DIR}/'!"
    )

    # Clean up temporary context file
    if os.path.exists("temporary_edit.json"):
        os.remove("temporary_edit.json")


def main():
    """Main CLI menu interface."""
    while True:
        print("\n=========================================")
        print("   OSINT Knowledge Graph Pipeline Menu")
        print("=========================================")
        print("[1] Extract Original Triples")
        print("[2] Generate Counterfactual Image (FLUX Pixels)")
        print("[3] Apply Graph Edit on Counterfactual (JSON Logic)")
        print("[4] Exit Program")
        print("=========================================")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            extract_original_triples()
            gc.collect()
            torch.cuda.empty_cache()

        elif choice == "2":
            run_image_generation()
            gc.collect()
            torch.cuda.empty_cache()

        elif choice == "3":
            generate_counterfactual_triples()
            gc.collect()
            torch.cuda.empty_cache()

        elif choice == "4":
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()