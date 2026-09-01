import json
import os
import re

def clean_term(text):
    # Καθαρισμός όρων από αριθμούς στην αρχή, παρενθέσεις και ειδικούς χαρακτήρες.
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip(" -.*\"'")

def clean_literal_value(predicate, value):
    # Καθαρισμός τιμών literals (π.χ. υπόλοιπα, κέρδη) από περιττό κείμενο ή παρενθέσεις.
    if not isinstance(value, str):
        return value
    
    p_lower = predicate.lower()
    
    if any(k in p_lower for k in ["balance", "profit", "loss"]):
        match = re.search(r'([$€£¥]?\s*[\d,]+(?:\.\d+)?)', value)
        if match:
            return match.group(1).strip()
            
    value = re.sub(r'\s*\b[A-Za-z\s]*\)$', '', value).strip()
    return value

def parse_triples(raw_output):
    # Αναλύει τις γραμμές του raw output του LLM και εξάγει triples στη μορφή (subject, predicate, object).
    extracted_data = []
    lines = raw_output.strip().split('\n')
    for line in lines:
        if '|' in line:
            parts = [clean_term(p) for p in line.split('|')]
            if len(parts) == 3 and all(parts):
                extracted_data.append({
                    "subject": parts[0],
                    "predicate": parts[1],
                    "object": parts[2]
                })
    return extracted_data

def save_description_txt(description, img_name, results_dir):
    # Αποθηκεύει τη λεκτική περιγραφή της εικόνας σε αρχείο .txt.
    name = os.path.splitext(img_name)[0]
    with open(os.path.join(results_dir, f"{name}_desc.txt"), 'w', encoding='utf-8') as f:
        f.write(description)

def save_json(data, filename, results_dir):
    # Αποθηκεύει τα δεδομένα του Knowledge Graph σε μορφή JSON.
    with open(os.path.join(results_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_and_deduplicate(triples):
    # Αφαιρεί διπλότυπα triples και φιλτράρει λέξεις-κλειδιά που προέρχονται από το prompt.
    unique_triples = []
    seen = set()
    prompt_keywords = ['format', 'constraint', 'predicate', 'example', 'triples', 'allowed']
    
    for t in triples:
        s, p, o = t["subject"], t["predicate"], t["object"]
        if any(k in s.lower() or k in p.lower() or k in o.lower() for k in prompt_keywords):
            continue
            
        triple_key = (s.lower(), p.lower(), o.lower())
        if triple_key not in seen:
            seen.add(triple_key)
            unique_triples.append(t)
            
    return unique_triples

def filter_ner_by_allowed_classes(ner_list, allowed_classes):
    # Φιλτράρει τις οντότητες NER βάσει των επιτρεπόμενων κλάσεων της οντολογίας.
    allowed_classes_lower = {str(c).strip().lower().replace('"', '').replace("'", '') for c in allowed_classes}
    
    allowed_ids = set()
    for ent in ner_list:
        pred = str(ent.get("predicate", "")).strip().lower()
        obj = str(ent.get("object", "")).strip().lower().replace('"', '').replace("'", '')
        
        if pred in ["rdf:type", "type"] and obj in allowed_classes_lower:
            allowed_ids.add(ent.get("subject"))

    return [ent for ent in ner_list if ent.get("subject") in allowed_ids]

def deduplicate_entities_by_label(ner_list):
    # Αφαιρεί διπλότυπες οντότητες και επαναριθμεί τα IDs (π.χ. Person_1, Person_2).
    seen = set()
    unique_entities = []
    
    for ent in ner_list:
        key = (ent.get("subject"), ent.get("predicate"), ent.get("object"))
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    id_map = {}
    class_counters = {}
    subjects = list(dict.fromkeys([e["subject"] for e in unique_entities]))

    for old_id in subjects:
        base_class = re.sub(r'_\d+$', '', old_id)
        
        if base_class not in class_counters:
            class_counters[base_class] = 1
        else:
            class_counters[base_class] += 1
            
        new_id = f"{base_class}_{class_counters[base_class]}"
        id_map[old_id] = new_id

    reindexed_entities = []
    for ent in unique_entities:
        new_ent = dict(ent)
        if new_ent["subject"] in id_map:
            new_ent["subject"] = id_map[new_ent["subject"]]
        reindexed_entities.append(new_ent)

    return reindexed_entities, id_map

def apply_dynamic_ner_filtering(img_name, ner_list):
    # Εφαρμόζει δυναμικό φιλτράρισμα οντοτήτων ανάλογα με τον τύπο της εικόνας (Viber, Profile, Dashboard).
    img_name_lower = img_name.lower()

    if "viber" in img_name_lower:
        allowed_viber_classes = {"profile_page", "person", "location"}
        return filter_ner_by_allowed_classes(ner_list, allowed_viber_classes)

    elif any(k in img_name_lower for k in ["fb", "facebook", "profile"]):
        allowed_profile_classes = {"profile_page", "person", "location", "organisation", "visual_symbol"}
        return filter_ner_by_allowed_classes(ner_list, allowed_profile_classes)

    elif any(k in img_name_lower for k in ["dashboard", "chart", "trading", "analytics"]):
        allowed_dashboard_classes = {"investment_account_page", "organisation", "person"}
        return filter_ner_by_allowed_classes(ner_list, allowed_dashboard_classes)

    return ner_list

def extract_allowed_classes_from_schema(schema_content):
    # Εξάγει τις επιτρεπόμενες κλάσεις από το κείμενο του schema (metapaths).
    allowed_classes = set()
    lines = schema_content.strip().split('\n')
    
    for line in lines:
        if '|' in line and not line.strip().startswith('#'):
            parts = [clean_term(p) for p in line.split('|')]
            if len(parts) == 3:
                allowed_classes.add(parts[0].strip().lower())
                allowed_classes.add(parts[2].strip().lower())
                
    return allowed_classes

def filter_relations_by_schema(raw_relations, extracted_entities, schema_content, id_map=None):
    # Ελέγχει και φιλτράρει τις σχέσεις ώστε να συμμορφώνονται αυστηρά με τους κανόνες του schema.
    if id_map is None:
        id_map = {}

    allowed_schema = {}
    lines = schema_content.strip().split('\n')
    for line in lines:
        if '|' in line and not line.strip().startswith('#'):
            parts = [clean_term(p) for p in line.split('|')]
            if len(parts) == 3:
                sub_type = parts[0].strip().lower()
                pred = parts[1].strip().lower()
                obj_type = parts[2].strip().lower()
                if pred not in allowed_schema:
                    allowed_schema[pred] = []
                allowed_schema[pred].append((sub_type, obj_type))

    entity_types = {}
    valid_entity_ids = set()
    
    for ent in extracted_entities:
        s = ent.get("subject", "").strip()
        p = ent.get("predicate", "").strip()
        o = ent.get("object", "").strip().replace('"', '')
        
        s_lower = s.lower()
        valid_entity_ids.add(s_lower)
        
        if p.lower() in ["rdf:type", "type"]:
            entity_types[s_lower] = o.lower()

    cleaned_relations = []
    seen_rel_keys = set()
    
    for rel in raw_relations:
        raw_s = rel.get("subject", "").strip()
        raw_o = rel.get("object", "").strip()
        
        s = id_map.get(raw_s, raw_s)
        o = id_map.get(raw_o, raw_o)
        p = rel.get("predicate", "").strip()
        
        s_lower = s.lower()
        p_lower = p.lower()
        o_lower = o.lower()

        literal_keywords = ["phone", "string", "literal", "balance", "profit", "loss", "role", "url", "code"]
        is_literal_relation = any(k in p_lower for k in literal_keywords)
        
        if s_lower not in valid_entity_ids:
            continue
            
        if not is_literal_relation and o_lower not in valid_entity_ids:
            continue

        if s_lower == o_lower or p_lower not in allowed_schema:
            continue

        raw_s_type = entity_types.get(s_lower, re.sub(r'_\d+$', '', s_lower))
        raw_o_type = entity_types.get(o_lower, re.sub(r'_\d+$', '', o_lower)) if not is_literal_relation else "string"

        match_found = False
        for valid_sub, valid_obj in allowed_schema[p_lower]:
            sub_match = (raw_s_type == valid_sub) or (valid_sub in raw_s_type) or (raw_s_type in valid_sub)
            obj_match = is_literal_relation or (raw_o_type == valid_obj) or (valid_obj in raw_o_type) or (raw_o_type in valid_obj)
            
            if sub_match and obj_match:
                match_found = True
                break
        
        if match_found:
            rel_key = (s_lower, p_lower, o_lower)
            if rel_key not in seen_rel_keys:
                seen_rel_keys.add(rel_key)
                cleaned_relations.append({
                    "subject": s,
                    "predicate": p,
                    "object": o
                })

    return cleaned_relations

def clean_graph_entities(ner_list, valid_relations, schema_content):
    # Καθαρίζει τις οντότητες διαγράφοντας όσες παραμένουν ασύνδετες στο τελικό γράφημα.
    allowed_classes = extract_allowed_classes_from_schema(schema_content)
    valid_schema_ids = set()

    for ent in ner_list:
        pred = str(ent.get("predicate", "")).lower()
        obj = str(ent.get("object", "")).lower().replace('"', '')
        subj = ent.get("subject")

        if pred in ["rdf:type", "type"] and obj in allowed_classes:
            valid_schema_ids.add(subj)

    connected_ids = {rel.get("subject") for rel in valid_relations} | {rel.get("object") for rel in valid_relations}

    final_entities = []
    for ent in ner_list:
        s = ent.get("subject")
        s_lower = s.lower() if s else ""

        if s in valid_schema_ids:
            if s in connected_ids or any(k in s_lower for k in ["page", "account", "interface", "profile"]):
                final_entities.append(ent)

    return final_entities, valid_relations

def process_pipeline_relations(raw_relations_str, ner_list, schema_content, id_map, img_name=""):
    # Εκτελεί τη συνολική ροή επεξεργασίας και καθαρισμού των σχέσεων του pipeline.
    relations_list = parse_triples(raw_relations_str)
    cleaned_relations = clean_and_deduplicate(relations_list)
    
    valid_relations = filter_relations_by_schema(
        cleaned_relations, 
        ner_list, 
        schema_content, 
        id_map=id_map
    )
    
    final_entities, valid_relations = clean_graph_entities(
        ner_list, 
        valid_relations, 
        schema_content
    )
    
    return final_entities, valid_relations

def apply_rdf_ontology_mapping(entities):
    # Αντιστοιχίζει τις βασικές κλάσεις οντοτήτων στα επίσημα URIs της DBpedia.
    class_mapping = {
        "location": "http://dbpedia.org/ontology/Location",
        "person": "http://dbpedia.org/ontology/Person",
        "organisation": "http://dbpedia.org/ontology/Organisation"
    }
    
    mapped_entities = []
    for ent in entities:
        pred = str(ent.get("predicate", ""))
        obj = str(ent.get("object", "")).strip().replace('"', '').replace("'", '')
        
        if pred.lower() in ["rdf:type", "type"]:
            obj_lower = obj.lower()
            if obj_lower in class_mapping:
                ent["object"] = class_mapping[obj_lower]
                
        mapped_entities.append(ent)
        
    return mapped_entities

def post_process_graph(graph_entry):
    # Εφαρμόζει κανόνες μετα-επεξεργασίας (π.χ. Viber φιλτράρισμα, διόρθωση labels/literals) στο Knowledge Graph.
    description = graph_entry.get("description", "").lower()
    img_name = graph_entry.get("image", "").lower()
    relations = graph_entry.get("relations", [])
    entities = graph_entry.get("entities", [])

    if "viber" in description or "viber" in img_name:
        relations = [
            rel for rel in relations 
            if not (str(rel.get("subject", "")).lower().startswith("person") 
                    and str(rel.get("predicate", "")).lower() in ["located_in", "originates_from"])
        ]

    clean_rels = []
    for rel in relations:
        obj_val = str(rel.get("object", "")).strip()
        if obj_val.lower() not in ["string", "none", "null", "xsd:string"]:
            pred = rel.get("predicate", "")
            rel["object"] = clean_literal_value(pred, obj_val)
            clean_rels.append(rel)

    subject_types = {}
    for ent in entities:
        s = ent.get("subject")
        p = str(ent.get("predicate")).lower()
        o = str(ent.get("object")).lower().replace('"', '').replace("'", '')
        if p in ["rdf:type", "type"]:
            subject_types[s] = o

    profile_acc_ids = [subj for subj, stype in subject_types.items() if any(k in stype for k in ["profile_page", "investment_account_page"])]
    person_ids = [subj for subj, stype in subject_types.items() if "person" in stype]
    
    if profile_acc_ids and person_ids:
        has_depicts = any(str(r.get("predicate", "")).lower() == "depicts_person" for r in clean_rels)
        if not has_depicts:
            clean_rels.append({
                "subject": profile_acc_ids[0],
                "predicate": "depicts_person",
                "object": person_ids[0]
            })

    processed_entities = []
    for ent in entities:
        subj = ent.get("subject")
        pred = str(ent.get("predicate")).strip()
        obj = str(ent.get("object")).strip()
        
        stype = subject_types.get(subj, "").lower()
        is_label_pred = pred.lower() in ["label", "rdfs:label"]

        if "investment_account_page" in stype and is_label_pred:
            continue

        if "profile_page" in stype and is_label_pred:
            processed_entities.append({
                "subject": subj,
                "predicate": "platform",
                "object": obj
            })
            continue

        if "location" in stype and is_label_pred and obj.startswith("http://dbpedia.org/"):
            processed_entities.append({
                "subject": subj,
                "predicate": "owl:sameAs",
                "object": obj
            })
            continue

        if pred.lower() == "label":
            pred = "rdfs:label"

        processed_entities.append({
            "subject": subj,
            "predicate": pred,
            "object": obj
        })

    graph_entry["entities"] = apply_rdf_ontology_mapping(processed_entities)
    graph_entry["relations"] = clean_rels
    return graph_entry