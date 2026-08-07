import os
import re
import json
import gc
import torch
import model
import utils
import prompts

RESULTS_DIR = 'results'
CF_RESULTS_DIR = 'counterfactual_results'
IMAGES_DIR = 'images'
CF_IMAGES_DIR = 'counterfactual'
SCHEMA_PATH = 'visual_metapath.txt' 
EXAMPLES_PATH = 'triples_example.txt'

def extract_original_triples():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    schema_content = ""
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()
    elif os.path.exists('prompts/schema.txt'):
        with open('prompts/schema.txt', 'r', encoding='utf-8') as f:
            schema_content = f.read()
    
    examples_content = ""
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, 'r', encoding='utf-8') as f:
            examples_content = f.read()

    image_list = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    for img_name in image_list:
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"\n==========================================")
        print(f"Processing Image: {img_name}")
        print(f"==========================================")
        
        # 1. Visual Description Stage
        description = model.generate_description(img_path, prompts.get_description_prompt())
        utils.save_description_txt(description, img_name, RESULTS_DIR)
        
        # 2. Named Entity Recognition (NER) Stage
        print(f"Extracting entities (NER) for {img_name}...")
        raw_ner_str = model.extract_entities(description, prompts.get_ner_prompt)
        
        print("\n--- RAW NER OUTPUT ---")
        print(raw_ner_str)
        print("----------------------")
        
        raw_ner_triples = utils.parse_triples(raw_ner_str)
        cleaned_ner = utils.clean_and_deduplicate(raw_ner_triples)
        
        # Dynamic NER Filtering (αν υπάρχει στο utils.py)
        if hasattr(utils, 'apply_dynamic_ner_filtering'):
            filtered_ner = utils.apply_dynamic_ner_filtering(img_name, cleaned_ner)
        else:
            allowed_classes = utils.extract_allowed_classes_from_schema(schema_content)
            filtered_ner = utils.filter_ner_by_allowed_classes(cleaned_ner, allowed_classes)
        
        # Deduplicate & Re-index IDs
        ner_list, id_map = utils.deduplicate_entities_by_label(filtered_ner)

        # 3. Relation Extraction (RE) Stage
        print(f"Extracting relations for {img_name}...")
        filtered_ner_str = "\n".join([f"{e['subject']} | {e['predicate']} | {e['object']}" for e in ner_list])
        
        raw_relations_str = model.extract_triples(
            description, 
            schema_content, 
            examples_content, 
            filtered_ner_str, 
            lambda d, s, e, n: prompts.get_relation_extraction_prompt(d, s, e, n)
        )
        
        # Process Relations & Entities
        final_entities, valid_relations = utils.process_pipeline_relations(
            raw_relations_str, 
            ner_list, 
            schema_content, 
            id_map,
            img_name=img_name
        )
        
        # 4. Construct Output Data Structure
        base_name = os.path.splitext(img_name)[0]
        graph_entry = {
            "image": img_name,
            "description": description,
            "entities": final_entities,
            "relations": valid_relations
        }
        
        # 5. Apply Post-Processing
        graph_entry = utils.post_process_graph(graph_entry)
        
        # 6. Save Final Graph JSON
        utils.save_json([graph_entry], f"{base_name}_graph.json", RESULTS_DIR)
        print(f"[SUCCESS] Saved original graph JSON for {img_name}")

def run_image_generation():
    if not os.path.exists(CF_IMAGES_DIR):
        os.makedirs(CF_IMAGES_DIR)

    image_list = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    if not image_list:
        print("No images found in images/ directory.")
        return
        
    img_name = image_list[0]
    base_name = os.path.splitext(img_name)[0]
    json_path = os.path.join(RESULTS_DIR, f"{base_name}_graph.json")
    
    if not os.path.exists(json_path):
        print(f"Original graph JSON not found for {img_name}. Run option [1] first.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)[0]

    entities = graph_data.get("entities", [])
    
    target_classes_priority = ['role', 'visual_symbol', 'account_name', 'organisation', 'location']
    
    target_entity = None
    for target_cls in target_classes_priority:
        for ent in entities:
            pred = str(ent.get("predicate", "")).lower()
            obj = str(ent.get("object", "")).lower().replace('"', '')
            subj = str(ent.get("subject", "")).lower()

            if (pred in ["rdf:type", "type"] and obj == target_cls) or subj.startswith(target_cls):
                target_entity = ent.get("subject")
                break
        if target_entity:
            break

    if not target_entity:
        for ent in entities:
            if ent.get("predicate") in ["label", "rdfs:label"]:
                target_entity = ent.get("subject")
                break

    if not target_entity:
        print("Could not find a suitable target entity for counterfactual modification.")
        return

    change_from_label = ""
    for ent in entities:
        if ent.get("subject") == target_entity and ent.get("predicate") in ["label", "rdfs:label"]:
            change_from_label = ent.get("object")
            break

    print(f"Target selected for modification: {target_entity} ('{change_from_label}')")
    
    change_to_label = model.generate_description([], prompts.get_geometric_counterfactual_prompt(change_from_label))
    print(f"LLM proposed alternative value: '{change_to_label}'")

    temp_edit = {
        "original_image": img_name,
        "target_entity": target_entity,
        "change_from": change_from_label,
        "change_to": change_to_label
    }
    with open('temporary_edit.json', 'w', encoding='utf-8') as f:
        json.dump(temp_edit, f, indent=4, ensure_ascii=False)

    print("Unloading Ollama VRAM to prepare for FLUX...")
    os.system("curl http://localhost:11434/api/generate -d '{\"model\": \"gemma3:12b\", \"keep_alive\": 0}' > /dev/null 2>&1")

    print("Loading FLUX Pipeline for Counterfactual Generation...")
    from pipeline_cf import CreateCounterfactualImage
    cf_generator = CreateCounterfactualImage()
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    cf_image = cf_generator.generate(change_from_label, change_to_label, img_path)
    
    cf_img_name = f"{base_name}_cf.png"
    cf_image.save(os.path.join(CF_IMAGES_DIR, cf_img_name))
    print(f"[SUCCESS] Counterfactual image saved inside '{CF_IMAGES_DIR}/{cf_img_name}'!")

def generate_counterfactual_triples():
    if not os.path.exists('temporary_edit.json'):
        print("No active counterfactual generation context found. Run option [2] first.")
        return

    with open('temporary_edit.json', 'r', encoding='utf-8') as f:
        temp_edit = json.load(f)

    img_name = temp_edit["original_image"]
    base_name = os.path.splitext(img_name)[0]
    
    with open(os.path.join(RESULTS_DIR, f"{base_name}_graph.json"), 'r', encoding='utf-8') as f:
        original_graph_data = json.load(f)[0]

    detected_change = f"The original asset '{temp_edit['change_from']}' was replaced by '{temp_edit['change_to']}'."
    
    raw_graph_lines = []
    for ent in original_graph_data["entities"]:
        raw_graph_lines.append(f"{ent['subject']} | {ent['predicate']} | {ent['object']}")
    for rel in original_graph_data["relations"]:
        raw_graph_lines.append(f"{rel['subject']} | {rel['predicate']} | {rel['object']}")
    original_graph_str = "\n".join(raw_graph_lines)

    print("Applying Graph Edit reasoning...")
    edit_instructions = model.extract_entities(original_graph_str, lambda g: prompts.get_counterfactual_prompt(g, detected_change))
    
    cf_entities = [dict(d) for d in original_graph_data["entities"]]
    cf_relations = [dict(d) for d in original_graph_data["relations"]]

    target_id = temp_edit["target_entity"]
    new_value = temp_edit["change_to"]

    updated = False
    for ent in cf_entities:
        if ent["subject"] == target_id and ent["predicate"] in ["label", "rdfs:label"]:
            ent["object"] = new_value
            updated = True
            break

    if updated:
        print(f"[EDIT APPLIED] Updated {target_id} label to '{new_value}'")
    else:
        print("[WARNING] Could not automatically apply graph edit to entities list.")

    formatted_output = [{
        "counterfactual_image": f"{base_name}_cf.png",
        "detected_change": detected_change,
        "entities": cf_entities,
        "relations": cf_relations
    }]

    utils.save_json(formatted_output, f'{base_name}_counterfactual_results.json', CF_RESULTS_DIR)
    print(f"[SUCCESS] Counterfactual JSON saved inside '{CF_RESULTS_DIR}/'!")
    
    if os.path.exists('temporary_edit.json'):
        os.remove('temporary_edit.json')

def main():
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
        
        if choice == '1': 
            extract_original_triples()
            gc.collect()
            torch.cuda.empty_cache()
            
        elif choice == '2': 
            run_image_generation()
            gc.collect()
            torch.cuda.empty_cache()
            
        elif choice == '3': 
            generate_counterfactual_triples()
            gc.collect()
            torch.cuda.empty_cache()
            
        elif choice == '4': 
            break
        else: 
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()