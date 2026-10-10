import gc
import json
import os
import re
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
    Εξάγει περιγραφές, οντότητες (NER) και σχέσεις από τις αρχικές εικόνες.
    Εφαρμόζει στοχευμένο Entity Grounding στα Pixels και συγχρονίζει την περιγραφή.
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
        
        # 1. Δημιουργία αρχικής περιγραφής εικόνας μέσω VLM
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
        
        # Εκτέλεση στοχευμένου Grounding στα Entities & λήψη τυχόν διορθώσεων
        graph_entry, grounded_changes = utils.post_process_graph(graph_entry, img_path=img_path)
        
        # Αν διορθώθηκε κάποιο Entity (π.χ. Mappy -> Mary), συγχρονίζεται στοχευμένα και το description
        if grounded_changes:
            synced_description = utils.sync_description_with_grounded_entities(description, grounded_changes)
            graph_entry["description"] = synced_description
            utils.save_description_txt(synced_description, img_name, RESULTS_DIR)

        utils.save_json([graph_entry], f"{base_name}_graph.json", RESULTS_DIR)
        print(f"[SUCCESS] Saved original graph JSON for {img_name}")

def run_image_generation():
    """
    Επιλέγει οντότητα-στόχο:
    - Αν η ΠΕΡΙΓΡΑΦΗ (description) περιέχει τη λέξη "Viber": ΑΠΟΚΛΕΙΣΤΙΚΑ το Όνομα (Person Name).
    - Για ΛΟΙΠΕΣ ΕΙΚΟΝΕΣ: Location -> Name Label.
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

    description_text = str(graph_data.get("description", "")).lower()
    entities = graph_data.get("entities", [])
    relations = graph_data.get("relations", [])

    clean_change_from = ""
    target_entity = None
    target_predicate = "rdfs:label"

    # =========================================================================
    # STEP A: TARGET SELECTION BASED ON DESCRIPTION CONTENT
    # =========================================================================
    if "viber" in description_text:
        # ΑΠΟΚΛΕΙΣΤΙΚΟΣ ΚΑΝΟΝΑΣ VIBER: Επιλογή ΜΟΝΟ του Person Name Label από το description
        for ent in entities:
            pred_lower = str(ent.get("predicate", "")).lower()
            subj_lower = str(ent.get("subject", "")).lower()
            if pred_lower in ["label", "rdfs:label"] and "person" in subj_lower:
                target_entity = ent.get("subject")
                target_predicate = ent.get("predicate")
                clean_change_from = str(ent.get("object", "")).replace('"', '').strip()
                print(f"[TARGET MATCHED - VIBER DESCRIPTION -> PERSON NAME ONLY] Target Name: '{clean_change_from}'")
                break
    else:
        # ΓΙΑ ΟΛΕΣ ΤΙΣ ΑΛΛΕΣ ΕΙΚΟΝΕΣ (FB, Dashboard): Location -> Name Label
        location_predicates = ["located_in", "originates_from", "associated_with_country", "associated_country"]
        for loc_pred in location_predicates:
            for rel in relations:
                pred_lower = str(rel.get("predicate", "")).lower()
                if pred_lower == loc_pred:
                    target_entity = rel.get("subject")
                    target_predicate = rel.get("predicate")
                    raw_obj = str(rel.get("object", ""))
                    clean_change_from = raw_obj.split("/")[-1].replace("_", " ") if "dbpedia.org" in raw_obj else raw_obj
                    print(f"[TARGET MATCHED - LOCATION ({loc_pred.upper()})] Target: '{clean_change_from}'")
                    break
            if clean_change_from:
                break

        if not clean_change_from:
            for ent in entities:
                if str(ent.get("predicate", "")).lower() in ["label", "rdfs:label"]:
                    target_entity = ent.get("subject")
                    target_predicate = ent.get("predicate")
                    clean_change_from = str(ent.get("object", "")).replace('"', '').strip()
                    print(f"[TARGET MATCHED - ENTITY LABEL] Target Name: '{clean_change_from}' ({target_predicate})")
                    break

    if not clean_change_from:
        print("Could not find a valid target for modification.")
        return

    # Καθαρισμός τυχόν προθεμάτων
    clean_change_from = re.sub(r'^(currently in|from|works at|lives in)\s+', '', clean_change_from, flags=re.IGNORECASE).strip()

    # Step B: Δημιουργία νέας τιμής με αυστηρούς γεωμετρικούς/χωρικούς κανόνες & Retry Validation
    print(f"[INFO] Generating counterfactual using Strict Geometric Rules for: '{clean_change_from}'...")
    
    target_len = len(clean_change_from)
    allowed_tolerance = 1
    min_len = max(1, target_len - allowed_tolerance)
    max_len = target_len + allowed_tolerance

    expected_words = [w for w in clean_change_from.split() if w]
    expected_initials = [w[0].upper() for w in expected_words]

    change_to_label = ""
    candidate = ""

    for attempt in range(5):
        strict_prompt = prompts.get_geometric_counterfactual_prompt(clean_change_from)
        
        if attempt > 0:
            strict_prompt += f"\n\nCRITICAL RETRY #{attempt+1}: Your previous output '{candidate}' was rejected. Ensure: 1) Same category & gender as '{clean_change_from}'. 2) Initials MUST be {' '.join(expected_initials)}. 3) Total length MUST be strictly between {min_len} and {max_len} characters."

        candidate = model.generate_description([], strict_prompt).strip(" \"'\n`")
        candidate = re.sub(r'```.*?\n|\n```', '', candidate).strip()
        
        candidate_words = candidate.split()
        candidate_initials = [w[0].upper() for w in candidate_words if w]
        
        has_correct_initials = (len(candidate_initials) == len(expected_initials) and 
                                candidate_initials == expected_initials)
        has_correct_length = (min_len <= len(candidate) <= max_len)
        
        if has_correct_initials and has_correct_length:
            change_to_label = candidate
            break
            
        print(f"[RETRY {attempt+1}] LLM generated '{candidate}' ({len(candidate)} chars, Initials: {candidate_initials}). Required: {min_len}-{max_len} chars, Initials: {expected_initials}.")

    if not change_to_label:
        change_to_label = candidate

    print(f"\n[TARGET SELECTED & CONSTRAINED]")
    print(f"Target Entity   : {target_entity}")
    print(f"Target Predicate: {target_predicate}")
    print(f"Change From     : '{clean_change_from}' ({target_len} chars)")
    print(f"Change To       : '{change_to_label}' ({len(change_to_label)} chars)\n")

    # Προσωρινή αποθήκευση
    temp_edit = {
        "original_image": img_name,
        "target_entity": target_entity,
        "target_predicate": target_predicate,
        "change_from": clean_change_from,
        "change_to": change_to_label
    }
    with open('temporary_edit.json', 'w', encoding='utf-8') as f:
        json.dump(temp_edit, f, indent=4, ensure_ascii=False)

    # Αποδέσμευση VRAM και παραγωγή εικόνας
    print("Unloading Ollama VRAM to prepare for FLUX...")
    os.system("curl -s http://localhost:11434/api/generate -d '{\"model\": \"gemma3:12b\", \"keep_alive\": 0}' > /dev/null 2>&1")

    print("Loading FLUX Pipeline for Counterfactual Generation...")
    from pipeline_cf import CreateCounterfactualImage
    cf_generator = CreateCounterfactualImage()
    
    img_path = os.path.join(IMAGES_DIR, img_name)
    cf_image = cf_generator.generate(clean_change_from, change_to_label, img_path)
    
    cf_img_name = f"{base_name}_cf.png"
    cf_output_path = os.path.join(CF_IMAGES_DIR, cf_img_name)
    cf_image.save(cf_output_path)
    print(f"[SUCCESS] Counterfactual image saved inside '{cf_output_path}'!")

    # Audit Highlights
    try:
        from highlight import CounterfactualHighlighter
        highlighter = CounterfactualHighlighter()

        orig_highlight_path = os.path.join(CF_IMAGES_DIR, f"{base_name}_orig_highlight.png")
        highlighter.generate_highlight(img_path=img_path, target_text=clean_change_from, output_path=orig_highlight_path)

        cf_highlight_path = os.path.join(CF_IMAGES_DIR, f"{base_name}_cf_highlight.png")
        highlighter.generate_highlight(img_path=cf_output_path, target_text=change_to_label, output_path=cf_highlight_path)
    except Exception as e:
        print(f"[HIGHLIGHT ERROR] Failed to generate visual audit overlay: {e}")

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
    target_predicate = temp_edit["target_predicate"]
    new_value = temp_edit["change_to"]

    clean_new_value = re.sub(r'^(currently in|from)\s+', '', new_value, flags=re.IGNORECASE).strip()

    updated = False

    # 1. Ενημέρωση τοποθεσίας (DBpedia URIs) αν η σχέση αφορά Location Predicate
    if target_predicate in ["originates_from", "associated_with_country", "associated_country", "located_in"]:
        formatted_uri = f"http://dbpedia.org/resource/{clean_new_value.title().replace(' ', '_')}"
        for rel in cf_relations:
            if rel.get("subject") == target_id and rel.get("predicate") == target_predicate:
                rel["object"] = formatted_uri
                updated = True
                print(f"[LOCATION RELATION EDIT APPLIED] Updated relation {target_id} -> {target_predicate} -> '{formatted_uri}'")
                break

    # 2. Ενημέρωση αντικειμένου στις λοιπές σχέσεις (Relations)
    if not updated:
        for rel in cf_relations:
            if rel.get("subject") == target_id and rel.get("predicate") == target_predicate:
                rel["object"] = clean_new_value
                updated = True
                print(f"[RELATION EDIT APPLIED] Updated relation {target_id} -> {target_predicate} -> '{clean_new_value}'")
                break

    # 3. Ενημέρωση ετικέτας της οντότητας (Label)
    if not updated:
        for ent in cf_entities:
            if ent.get("subject") == target_id and ent.get("predicate") in ["label", "rdfs:label"]:
                ent["object"] = clean_new_value
                updated = True
                print(f"[ENTITY EDIT APPLIED] Updated entity {target_id} label to '{clean_new_value}'")
                break

    # Fallback εφαρμογή αλλαγής αν δεν εντοπίστηκε ακριβής κανόνας
    if not updated:
        print("[WARNING] Applying fallback to target entity label.")
        for ent in cf_entities:
            if ent.get("subject") == target_id:
                ent["object"] = clean_new_value
                break

    # Αποθήκευση τελικού τροποποιημένου γράφου
    if not os.path.exists(CF_RESULTS_DIR):
        os.makedirs(CF_RESULTS_DIR)

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