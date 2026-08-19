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
        
        description = model.generate_description(img_path, prompts.get_description_prompt())
        utils.save_description_txt(description, img_name, RESULTS_DIR)
        
        print(f"Extracting entities (NER) for {img_name}...")
        raw_ner_str = model.extract_entities(description, prompts.get_ner_prompt)
        
        raw_ner_triples = utils.parse_triples(raw_ner_str)
        cleaned_ner = utils.clean_and_deduplicate(raw_ner_triples)
        
        if hasattr(utils, 'apply_dynamic_ner_filtering'):
            filtered_ner = utils.apply_dynamic_ner_filtering(img_name, cleaned_ner)
        else:
            allowed_classes = utils.extract_allowed_classes_from_schema(schema_content)
            filtered_ner = utils.filter_ner_by_allowed_classes(cleaned_ner, allowed_classes)
        
        ner_list, id_map = utils.deduplicate_entities_by_label(filtered_ner)

        print(f"Extracting relations for {img_name}...")
        filtered_ner_str = "\n".join([f"{e['subject']} | {e['predicate']} | {e['object']}" for e in ner_list])
        
        raw_relations_str = model.extract_triples(
            description, 
            schema_content, 
            examples_content, 
            filtered_ner_str, 
            lambda d, s, e, n: prompts.get_relation_extraction_prompt(d, s, e, n)
        )
        
        final_entities, valid_relations = utils.process_pipeline_relations(
            raw_relations_str, 
            ner_list, 
            schema_content, 
            id_map,
            img_name=img_name
        )
        
        base_name = os.path.splitext(img_name)[0]
        graph_entry = {
            "image": img_name,
            "description": description,
            "entities": final_entities,
            "relations": valid_relations
        }
        
        graph_entry = utils.post_process_graph(graph_entry)
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
    relations = graph_data.get("relations", [])
    
    target_entity = None
    change_from_label = ""
    target_predicate = ""

    # 1. ΠΡΩΤΗ ΠΡΟΤΕΡΑΙΟΤΗΤΑ: Ψάχνουμε 'has_role' στα relations
    for rel in relations:
        p_lower = str(rel.get("predicate", "")).lower()
        if p_lower == "has_role":
            target_entity = rel.get("subject")
            target_predicate = rel.get("predicate")
            change_from_label = str(rel.get("object", ""))
            print(f"[ROLE MATCH] Found explicit role triple: {target_entity} -> has_role -> '{change_from_label}'")
            break

    # 2. FALLBACK: Αν δεν βρεθεί 'has_role', ψάχνουμε στις οντότητες
    if not target_entity:
        print("[INFO] No 'has_role' predicate found in relations. Falling back to default entity selection.")
        target_classes_priority = ['person', 'location', 'visual_symbol', 'organisation']
        
        for target_cls in target_classes_priority:
            for ent in entities:
                pred = str(ent.get("predicate", "")).lower()
                obj = str(ent.get("object", "")).lower().replace('"', '')
                subj = str(ent.get("subject", "")).lower()

                if (pred in ["rdf:type", "type"] and target_cls in obj) or subj.startswith(target_cls):
                    target_entity = ent.get("subject")
                    break
            if target_entity:
                break

        if target_entity:
            for ent in entities:
                p_lower = str(ent.get("predicate", "")).lower()
                if ent.get("subject") == target_entity and p_lower in ["label", "rdfs:label", "owl:sameas"]:
                    change_from_label = str(ent.get("object", ""))
                    target_predicate = ent.get("predicate")
                    break

    if not target_entity or not change_from_label:
        print("Could not find a valid entity or role for modification.")
        return

    clean_change_from = change_from_label.replace('"', '').strip()

    print(f"Target selected for modification: {target_entity} [{target_predicate}] -> '{clean_change_from}'")
    
    change_to_label = model.generate_description([], prompts.get_geometric_counterfactual_prompt(clean_change_from))
    change_to_label = change_to_label.strip(" \"'")
    print(f"LLM proposed alternative asset: '{change_to_label}'")

    temp_edit = {
        "original_image": img_name,
        "target_entity": target_entity,
        "target_predicate": target_predicate,
        "change_from": clean_change_from,
        "change_to": change_to_label
    }
    with open('temporary_edit.json', 'w', encoding='utf-8') as f:
        json.dump(temp_edit, f, indent=4, ensure_ascii=False)

    print("Unloading Ollama VRAM to prepare for FLUX...")
    os.system("curl -s http://localhost:11434/api/generate -d '{\"model\": \"gemma3:12b\", \"keep_alive\": 0}' > /dev/null 2>&1")

    print("Loading FLUX Pipeline for Counterfactual Generation...")
    from pipeline_cf import CreateCounterfactualImage
    cf_generator = CreateCounterfactualImage()
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    cf_image = cf_generator.generate(clean_change_from, change_to_label, img_path)
    
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
    
    cf_entities = [dict(d) for d in original_graph_data["entities"]]
    cf_relations = [dict(d) for d in original_graph_data["relations"]]

    target_id = temp_edit["target_entity"]
    target_pred = temp_edit["target_predicate"]
    new_value = temp_edit["change_to"]

    updated = False

    # 1. ΠΡΩΤΗ ΠΡΟΤΕΡΑΙΟΤΗΤΑ: Ψάχνουμε στα RELATIONS (π.χ. Person_1 -> has_role -> 'British Soldier')
    for rel in cf_relations:
        if rel["subject"] == target_id and rel["predicate"].lower() == target_pred.lower():
            rel["object"] = new_value
            updated = True
            print(f"[RELATION EDIT APPLIED] Updated relation {target_id} -> {target_pred} -> '{new_value}'")
            break

    # 2. SECONDARY: Αν δεν βρέθηκε στα relations, ψάχνουμε στα ENTITIES (π.χ. rdfs:label)
    if not updated:
        for ent in cf_entities:
            if ent["subject"] == target_id and ent["predicate"] in ["label", "rdfs:label"]:
                ent["object"] = new_value
                updated = True
                print(f"[ENTITY EDIT APPLIED] Updated entity {target_id} label to '{new_value}'")
                break

    if not updated:
        print("[WARNING] Could not automatically apply graph edit.")

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
