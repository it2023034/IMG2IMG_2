import os
import re
import json
import gc
import torch
import model
import utils
import prompts

# Ορισμός βασικών καταλόγων και διαδρομών αρχείων
RESULTS_DIR = 'results'
CF_RESULTS_DIR = 'counterfactual_results'
IMAGES_DIR = 'images'
CF_IMAGES_DIR = 'counterfactual'
SCHEMA_PATH = 'visual_metapath.txt' 
EXAMPLES_PATH = 'triples_example.txt'

def extract_original_triples():
    """
    Εξάγει περιγραφές, οντότητες (NER) και σχέσεις από τις αρχικές εικόνες
    και αποθηκεύει τα αποτελέσματα σε μορφή Knowledge Graph JSON.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Φόρτωση σχήματος (schema) αν υπάρχει
    schema_content = ""
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_content = f.read()
    elif os.path.exists('prompts/schema.txt'):
        with open('prompts/schema.txt', 'r', encoding='utf-8') as f:
            schema_content = f.read()
    
    # Φόρτωση παραδειγμάτων (examples) αν υπάρχουν
    examples_content = ""
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, 'r', encoding='utf-8') as f:
            examples_content = f.read()

    # Συλλογή και ταξινόμηση διαθέσιμων εικόνων
    image_list = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    for img_name in image_list:
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"\n==========================================")
        print(f"Processing Image: {img_name}")
        print(f"==========================================")
        
        # 1. Δημιουργία περιγραφής εικόνας μέσω LLM
        description = model.generate_description(img_path, prompts.get_description_prompt())
        utils.save_description_txt(description, img_name, RESULTS_DIR)
        
        # 2. Εξαγωγή και καθαρισμός οντοτήτων (NER)
        print(f"Extracting entities (NER) for {img_name}...")
        raw_ner_str = model.extract_entities(description, prompts.get_ner_prompt)
        
        raw_ner_triples = utils.parse_triples(raw_ner_str)
        cleaned_ner = utils.clean_and_deduplicate(raw_ner_triples)
        
        # Φιλτράρισμα οντοτήτων βάσει επιτρεπόμενων κλάσεων
        if hasattr(utils, 'apply_dynamic_ner_filtering'):
            filtered_ner = utils.apply_dynamic_ner_filtering(img_name, cleaned_ner)
        else:
            allowed_classes = utils.extract_allowed_classes_from_schema(schema_content)
            filtered_ner = utils.filter_ner_by_allowed_classes(cleaned_ner, allowed_classes)
        
        ner_list, id_map = utils.deduplicate_entities_by_label(filtered_ner)

        # 3. Εξαγωγή σχέσεων μεταξύ των οντοτήτων
        print(f"Extracting relations for {img_name}...")
        filtered_ner_str = "\n".join([f"{e['subject']} | {e['predicate']} | {e['object']}" for e in ner_list])
        
        raw_relations_str = model.extract_triples(
            description, 
            schema_content, 
            examples_content, 
            filtered_ner_str, 
            lambda d, s, e, n: prompts.get_relation_extraction_prompt(d, s, e, n)
        )
        
        # Επεξεργασία και επικύρωση σχέσεων
        final_entities, valid_relations = utils.process_pipeline_relations(
            raw_relations_str, 
            ner_list, 
            schema_content, 
            id_map,
            img_name=img_name
        )
        
        # 4. Αποθήκευση του τελικού Knowledge Graph
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
    """
    Επιλέγει μια οντότητα-στόχο, παράγει τροποποιημένο κείμενο βάσει κανόνων
    και δημιουργεί μια νέα παραλλαγμένη εικόνα (Counterfactual Image) μέσω FLUX.
    """
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

    description = graph_data.get("description", "")
    entities = graph_data.get("entities", [])

    # Step A: Επιλογή Target Entity/Value από το LLM
    cf_select_prompt = f"""
    Analyze the following image description and extracted entities.
    Select ONE specific target entity or visual attribute string (e.g. Person Name, Location, or Role) to modify.

    Description: "{description}"
    Extracted Entities: {json.dumps(entities)}

    Return ONLY a JSON object with keys "change_from", "target_entity", and "target_predicate":
    {{
        "change_from": "<exact string to modify, e.g. Michael Anderson>",
        "target_entity": "<subject ID, e.g. Person_1>",
        "target_predicate": "<predicate, e.g. label or rdfs:label>"
    }}
    """

    print("[INFO] Selecting target entity via LLM...")
    llm_response = model.generate_description([], cf_select_prompt)
    
    clean_change_from = ""
    target_entity = None
    target_predicate = "label"

    # Ανάλυση της απόκρισης JSON του LLM με fallback σε περίπτωση αποτυχίας
    try:
        json_clean = re.sub(r'```json\s*|\s*```', '', llm_response).strip()
        cf_target_data = json.loads(json_clean)
        clean_change_from = cf_target_data.get("change_from", "").replace('"', '').strip()
        target_entity = cf_target_data.get("target_entity", None)
        target_predicate = cf_target_data.get("target_predicate", "label")
    except Exception as e:
        print(f"[FALLBACK] Parsing failed ({e}). Extracting from graph labels...")
        for ent in entities:
            if str(ent.get("predicate", "")).lower() in ["label", "rdfs:label"]:
                target_entity = ent.get("subject")
                target_predicate = ent.get("predicate")
                clean_change_from = str(ent.get("object", "")).replace('"', '').strip()
                break

    if not clean_change_from:
        print("Could not find a valid target for modification.")
        return

    # Step B: Δημιουργία νέας τιμής με αυστηρούς γεωμετρικούς/χωρικούς κανόνες
    print(f"[INFO] Generating counterfactual using Strict Geometric Rules for: '{clean_change_from}'...")
    
    strict_prompt = prompts.get_geometric_counterfactual_prompt(clean_change_from)
    change_to_label = model.generate_description([], strict_prompt).strip(" \"'\n`")

    print(f"\n[TARGET SELECTED & CONSTRAINED]")
    print(f"Target Entity : {target_entity}")
    print(f"Change From   : '{clean_change_from}'")
    print(f"Change To     : '{change_to_label}'\n")

    # Προσωρινή αποθήκευση των δεδομένων αλλαγής για το επόμενο βήμα
    temp_edit = {
        "original_image": img_name,
        "target_entity": target_entity,
        "target_predicate": target_predicate,
        "change_from": clean_change_from,
        "change_to": change_to_label
    }
    with open('temporary_edit.json', 'w', encoding='utf-8') as f:
        json.dump(temp_edit, f, indent=4, ensure_ascii=False)

    # Αποδέσμευση μνήμης VRAM από το Ollama πριν τη φόρτωση του FLUX
    print("Unloading Ollama VRAM to prepare for FLUX...")
    os.system("curl -s http://localhost:11434/api/generate -d '{\"model\": \"gemma3:12b\", \"keep_alive\": 0}' > /dev/null 2>&1")

    # Παραγωγή παραλλαγμένης εικόνας (FLUX Pipeline)
    print("Loading FLUX Pipeline for Counterfactual Generation...")
    from pipeline_cf import CreateCounterfactualImage
    cf_generator = CreateCounterfactualImage()
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    cf_image = cf_generator.generate(clean_change_from, change_to_label, img_path)
    
    cf_img_name = f"{base_name}_cf.png"
    cf_image.save(os.path.join(CF_IMAGES_DIR, cf_img_name))
    print(f"[SUCCESS] Counterfactual image saved inside '{CF_IMAGES_DIR}/{cf_img_name}'!")

def generate_counterfactual_triples():
    """
    Ενημερώνει τη δομή του Knowledge Graph (JSON) ώστε να αντικατοπτρίζει 
    τις αλλαγές που πραγματοποιήθηκαν στην παραλλαγμένη εικόνα.
    """
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
    change_from = temp_edit["change_from"].lower()
    new_value = temp_edit["change_to"]

    clean_new_value = re.sub(r'^(currently in|from)\s+', '', new_value, flags=re.IGNORECASE).strip()

    updated = False

    # 1. Ενημέρωση τοποθεσίας (DBpedia URIs)
    for ent in cf_entities:
        obj_val = str(ent.get("object", "")).lower()
        if ent.get("predicate") == "owl:sameAs" and any(loc in change_from for loc in obj_val.split('/')[-1].lower().split('_')):
            formatted_uri = f"http://dbpedia.org/resource/{clean_new_value.replace(' ', '_')}"
            ent["object"] = formatted_uri
            updated = True
            print(f"[LOCATION EDIT APPLIED] Updated entity {ent['subject']} owl:sameAs to '{formatted_uri}'")
            break

    # 2. Ενημέρωση αντικειμένου στις σχέσεις (Relations)
    if not updated:
        for rel in cf_relations:
            if target_id and rel["subject"] == target_id:
                rel_obj = str(rel["object"]).lower()
                if rel_obj in change_from or change_from in rel_obj:
                    rel["object"] = clean_new_value
                    updated = True
                    print(f"[RELATION EDIT APPLIED] Updated relation {rel['subject']} -> {rel['predicate']} -> '{clean_new_value}'")
                    break

    # 3. Ενημέρωση ετικέτας της οντότητας (Label)
    if not updated:
        for ent in cf_entities:
            if target_id and ent["subject"] == target_id and ent["predicate"] in ["label", "rdfs:label"]:
                ent["object"] = clean_new_value
                updated = True
                print(f"[ENTITY EDIT APPLIED] Updated entity {target_id} label to '{clean_new_value}'")
                break

    # Fallback εφαρμογή αλλαγής αν δεν εντοπίστηκε ακριβής κανόνας
    if not updated:
        print("[WARNING] Could not automatically map the edit to the graph. Applying fallback to target entity label.")
        for ent in cf_entities:
            if target_id and ent["subject"] == target_id:
                ent["object"] = clean_new_value
                break

    # Αποθήκευση τελικού τροποποιημένου γράφου
    formatted_output = [{
        "counterfactual_image": f"{base_name}_cf.png",
        "detected_change": detected_change,
        "entities": cf_entities,
        "relations": cf_relations
    }]

    utils.save_json(formatted_output, f'{base_name}_counterfactual_results.json', CF_RESULTS_DIR)
    print(f"[SUCCESS] Counterfactual JSON saved inside '{CF_RESULTS_DIR}/'!")
    
    # Διαγραφή του προσωρινού αρχείου
    if os.path.exists('temporary_edit.json'):
        os.remove('temporary_edit.json')

def main():
    """Κεντρικό μενού αλληλεπίδρασης χρήστη."""
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