import json
import os
import re
from difflib import SequenceMatcher
import phonenumbers
from phonenumbers import geocoder
import easyocr

# Singleton αρχικοποίηση EasyOCR (Lazy loading)
_ocr_reader = None


def get_ocr_reader():
    """Επιστρέφει ένα singleton instance του EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


def ground_text_to_image_pixels(
    extracted_text: str, img_path: str, threshold: float = 0.65
) -> str:
    """GENERIC GROUNDING: Συγκρίνει το κείμενο που εξήγαγε το VLM με τα πραγματικά pixels 
    της εικόνας μέσω OCR. Αν βρει κείμενο στην εικόνα με ομοιότητα > threshold, 
    επιστρέφει την ΑΚΡΙΒΗ γραφή των pixels.
    """
    if (
        not extracted_text
        or not isinstance(extracted_text, str)
        or len(extracted_text.strip()) < 2
    ):
        return extracted_text

    if not os.path.exists(img_path):
        return extracted_text

    try:
        reader = get_ocr_reader()
        results = reader.readtext(img_path)

        target_clean = extracted_text.strip().lower()
        best_match = extracted_text
        best_score = 0.0

        for bbox, text, prob in results:
            found_clean = text.strip().lower()

            # Απευθείας ταύτιση -> Επιστρέφει την ακριβή γραφή των pixels
            if target_clean == found_clean:
                return text.strip()

            # Fuzzy match για μικρολάθη του VLM/OCR
            score = SequenceMatcher(None, target_clean, found_clean).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = text.strip()

        if best_score >= threshold:
            print(
                f"[GENERIC GROUNDING FIX] VLM entity label '{extracted_text}' corrected to exact image pixels: '{best_match}'"
            )
            return best_match

    except Exception as e:
        print(f"[GROUNDING WARNING] Could not perform OCR grounding on text: {e}")

    return extracted_text


def sync_description_with_grounded_entities(
    description: str, grounded_changes: dict
) -> str:
    """TARGETED DESC SYNC: Αντικαθιστά στο description ΜΟΝΟ τις συγκεκριμένες 
    λανθασμένες συμβολοσειρές που διορθώθηκαν κατά το Entity Grounding (π.χ. 'Mappy' -> 'Mary').
    """
    if not description or not grounded_changes:
        return description

    updated_desc = description
    for old_val, new_val in grounded_changes.items():
        if old_val and new_val and old_val != new_val:
            pattern = r"\b" + re.escape(old_val) + r"\b"
            updated_desc = re.sub(
                pattern, new_val, updated_desc, flags=re.IGNORECASE
            )
            print(
                f"[DESC SYNC] Replaced '{old_val}' with '{new_val}' in description."
            )

    return updated_desc


def clean_term(text):
    """Cleans an extracted term by removing leading numbering, parenthetical noise, and quotes."""
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    return text.strip(" -.*\"'")


def clean_literal_value(predicate, value):
    """Sanitizes literal attribute values (e.g., balance, profit) from redundant text."""
    if not isinstance(value, str):
        return value

    p_lower = predicate.lower()

    if any(k in p_lower for k in ["balance", "profit", "loss"]):
        match = re.search(r"([$€£¥]?\s*[\d,]+(?:\.\d+)?)", value)
        if match:
            return match.group(1).strip()

    value = re.sub(r"\s*\b[A-Za-z\s]*\)$", "", value).strip()
    return value


def parse_triples(raw_output):
    """Parses raw pipe-delimited text output from LLMs into structured triple dictionaries."""
    extracted_data = []
    lines = raw_output.strip().split("\n")
    for line in lines:
        if "|" in line:
            parts = [clean_term(p) for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                extracted_data.append(
                    {
                        "subject": parts[0],
                        "predicate": parts[1],
                        "object": parts[2],
                    }
                )
    return extracted_data


def save_description_txt(description, img_name, results_dir):
    """Saves the generated text description of an image to a text file."""
    name = os.path.splitext(img_name)[0]
    with open(
        os.path.join(results_dir, f"{name}_desc.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(description)


def save_json(data, filename, results_dir):
    """Saves structured Knowledge Graph data into a formatted JSON file."""
    with open(os.path.join(results_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def clean_and_deduplicate(triples):
    """Deduplicates extracted triples and filters out prompt artifact keywords."""
    unique_triples = []
    seen = set()
    prompt_keywords = [
        "format",
        "constraint",
        "predicate",
        "example",
        "triples",
        "allowed",
    ]

    for t in triples:
        s, p, o = t["subject"], t["predicate"], t["object"]
        if any(
            k in s.lower() or k in p.lower() or k in o.lower()
            for k in prompt_keywords
        ):
            continue

        triple_key = (s.lower(), p.lower(), o.lower())
        if triple_key not in seen:
            seen.add(triple_key)
            unique_triples.append(t)

    return unique_triples


def filter_ner_by_allowed_classes(ner_list, allowed_classes):
    allowed_classes_lower = {
        str(c).strip().lower().replace('"', "").replace("'", "")
        for c in allowed_classes
    }

    allowed_ids = set()
    for ent in ner_list:
        pred = str(ent.get("predicate", "")).strip().lower()
        obj = (
            str(ent.get("object", ""))
            .strip()
            .lower()
            .replace('"', "")
            .replace("'", "")
        )
        subj = str(ent.get("subject", "")).strip()

        if pred in ["rdf:type", "type"] and obj in allowed_classes_lower:
            allowed_ids.add(subj)
        elif any(c in subj.lower() for c in allowed_classes_lower):
            allowed_ids.add(subj)

    return [ent for ent in ner_list if ent.get("subject") in allowed_ids]


def deduplicate_entities_by_label(ner_list):
    """Deduplicates entity IDs and reindexes them sequentially per class starting strictly from 1."""
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
        base_class = re.sub(r"_\d+$", "", old_id)

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
    """Applies dynamic entity filtering rules without standalone Location nodes."""
    img_name_lower = img_name.lower()

    if "viber" in img_name_lower:
        allowed_viber_classes = {"profile_page", "person"}
        return filter_ner_by_allowed_classes(ner_list, allowed_viber_classes)

    elif any(k in img_name_lower for k in ["fb", "facebook", "profile"]):
        allowed_profile_classes = {
            "profile_page",
            "person",
            "organisation",
            "visual_symbol",
        }
        return filter_ner_by_allowed_classes(ner_list, allowed_profile_classes)

    elif any(
        k in img_name_lower
        for k in ["dashboard", "chart", "trading", "analytics"]
    ):
        allowed_dashboard_classes = {
            "investment_account_page",
            "organisation",
            "person",
        }
        return filter_ner_by_allowed_classes(
            ner_list, allowed_dashboard_classes
        )

    return ner_list


def extract_allowed_classes_from_schema(schema_content):
    """Extracts valid domain and range classes from schema metapaths."""
    allowed_classes = set()
    lines = schema_content.strip().split("\n")

    for line in lines:
        if "|" in line and not line.strip().startswith("#"):
            parts = [clean_term(p) for p in line.split("|")]
            if len(parts) == 3:
                allowed_classes.add(parts[0].strip().lower())
                allowed_classes.add(parts[2].strip().lower())

    return allowed_classes


def filter_relations_by_schema(
    raw_relations, extracted_entities, schema_content, id_map=None, img_name=""
):
    """Filters and validates relation triples against strict domain/range schema constraints."""
    if id_map is None:
        id_map = {}

    allowed_schema = {}
    lines = schema_content.strip().split("\n")
    for line in lines:
        if "|" in line and not line.strip().startswith("#"):
            parts = [clean_term(p) for p in line.split("|")]
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
        o = ent.get("object", "").strip().replace('"', "")

        s_lower = s.lower()
        valid_entity_ids.add(s_lower)

        if p.lower() in ["rdf:type", "type"]:
            entity_types[s_lower] = o.lower()

    cleaned_relations = []
    seen_rel_keys = set()
    img_name_lower = img_name.lower()

    viber_allowed_predicates = {
        "has_phone_number",
        "originates_from",
        "depicts_person",
    }

    for rel in raw_relations:
        raw_s = rel.get("subject", "").strip()
        raw_o = rel.get("object", "").strip()

        s = id_map.get(raw_s, raw_s)
        o = id_map.get(raw_o, raw_o)
        p = rel.get("predicate", "").strip()

        s_lower = s.lower()
        p_lower = p.lower()
        o_lower = o.lower()

        if "viber" in img_name_lower and p_lower not in viber_allowed_predicates:
            continue

        external_uri_predicates = ["associated_with_country", "located_in"]
        literal_keywords = [
            "phone",
            "string",
            "literal",
            "balance",
            "profit",
            "loss",
            "role",
            "url",
            "code",
        ]

        is_direct_value = any(k in p_lower for k in literal_keywords) or (
            p_lower in external_uri_predicates
        )

        if s_lower not in valid_entity_ids:
            continue

        if not is_direct_value and o_lower not in valid_entity_ids:
            continue

        if s_lower == o_lower or p_lower not in allowed_schema:
            continue

        raw_s_type = entity_types.get(s_lower, re.sub(r"_\d+$", "", s_lower))
        raw_o_type = (
            entity_types.get(o_lower, re.sub(r"_\d+$", "", o_lower))
            if not is_direct_value
            else "location"
            if p_lower in external_uri_predicates
            else "string"
        )

        match_found = False
        for valid_sub, valid_obj in allowed_schema[p_lower]:
            sub_match = (
                (raw_s_type == valid_sub)
                or (valid_sub in raw_s_type)
                or (raw_s_type in valid_sub)
            )
            obj_match = (
                is_direct_value
                or (raw_o_type == valid_obj)
                or (valid_obj in raw_o_type)
                or (raw_o_type in valid_obj)
            )

            if sub_match and obj_match:
                match_found = True
                break

        if match_found:
            rel_key = (s_lower, p_lower, o_lower)
            if rel_key not in seen_rel_keys:
                seen_rel_keys.add(rel_key)
                cleaned_relations.append(
                    {"subject": s, "predicate": p, "object": o}
                )

    return cleaned_relations


def clean_graph_entities(ner_list, valid_relations, schema_content):
    """Removes disconnected entities that participate in no valid relations."""
    allowed_classes = extract_allowed_classes_from_schema(schema_content)
    valid_schema_ids = set()

    for ent in ner_list:
        pred = str(ent.get("predicate", "")).lower()
        obj = str(ent.get("object", "")).lower().replace('"', "")
        subj = ent.get("subject")

        if pred in ["rdf:type", "type"] and obj in allowed_classes:
            valid_schema_ids.add(subj)

    connected_ids = {rel.get("subject") for rel in valid_relations} | {
        rel.get("object") for rel in valid_relations
    }

    final_entities = []
    for ent in ner_list:
        s = ent.get("subject")
        s_lower = s.lower() if s else ""

        if s in valid_schema_ids:
            if s in connected_ids or any(
                k in s_lower
                for k in ["page", "account", "interface", "profile"]
            ):
                final_entities.append(ent)

    return final_entities, valid_relations


def process_pipeline_relations(
    raw_relations_str, ner_list, schema_content, id_map, img_name=""
):
    """Executes the full relation processing, cleaning, and filtering pipeline."""
    relations_list = parse_triples(raw_relations_str)
    cleaned_relations = clean_and_deduplicate(relations_list)

    valid_relations = filter_relations_by_schema(
        cleaned_relations,
        ner_list,
        schema_content,
        id_map=id_map,
        img_name=img_name,
    )

    final_entities, valid_relations = clean_graph_entities(
        ner_list, valid_relations, schema_content
    )

    return final_entities, valid_relations


def apply_rdf_ontology_mapping(entities):
    """Maps core entity classes to official DBpedia ontology URIs."""
    class_mapping = {
        "person": "http://dbpedia.org/ontology/Person",
        "organisation": "http://dbpedia.org/ontology/Organisation",
    }

    mapped_entities = []
    for ent in entities:
        pred = str(ent.get("predicate", ""))
        obj = str(ent.get("object", "")).strip().replace('"', "").replace("'", "")

        if pred.lower() in ["rdf:type", "type"]:
            obj_lower = obj.lower()
            if obj_lower in class_mapping:
                ent["object"] = class_mapping[obj_lower]

        mapped_entities.append(ent)

    return mapped_entities


def post_process_graph(graph_entry, img_path=""):
    """Applies clean post-processing rules to final Knowledge Graph entities and relations."""
    relations = graph_entry.get("relations", [])
    entities = graph_entry.get("entities", [])
    grounded_changes = {}  # Καταγραφή των αλλαγών (old_val -> new_val)

    # =========================================================================
    # 1. FORCE SEQUENTIAL RE-INDEXING (e.g., Organisation_2 -> Organisation_1)
    # =========================================================================
    id_map = {}
    class_counters = {}

    for ent in entities:
        subj = ent.get("subject")
        if subj and subj not in id_map:
            base_class = re.sub(r"_\d+$", "", subj)
            class_counters[base_class] = class_counters.get(base_class, 0) + 1
            id_map[subj] = f"{base_class}_{class_counters[base_class]}"

    for ent in entities:
        if ent.get("subject") in id_map:
            ent["subject"] = id_map[ent["subject"]]

    for rel in relations:
        if rel.get("subject") in id_map:
            rel["subject"] = id_map[rel["subject"]]
        if rel.get("object") in id_map:
            rel["object"] = id_map[rel["object"]]

    # =========================================================================
    # 2. CLEAN & FILTER RELATIONS (Remove 'N/A', 'null', 'none', etc.)
    # =========================================================================
    clean_rels = []
    invalid_literals = {
        "string",
        "none",
        "null",
        "xsd:string",
        "n/a",
        "unknown",
        "",
    }

    for rel in relations:
        obj_val = str(rel.get("object", "")).strip()
        if obj_val.lower() not in invalid_literals:
            pred = rel.get("predicate", "")
            rel["object"] = clean_literal_value(pred, obj_val)
            clean_rels.append(rel)

    subject_types = {}
    for ent in entities:
        s = ent.get("subject")
        p = str(ent.get("predicate")).lower()
        o = str(ent.get("object")).lower().replace('"', "").replace("'", "")
        if p in ["rdf:type", "type"]:
            subject_types[s] = o

    profile_acc_ids = [
        subj
        for subj, stype in subject_types.items()
        if any(k in stype for k in ["profile_page", "investment_account_page"])
    ]
    person_ids = [
        subj for subj, stype in subject_types.items() if "person" in stype
    ]

    # Ensure container page depicts the primary person
    if profile_acc_ids and person_ids:
        has_depicts = any(
            str(r.get("predicate", "")).lower() == "depicts_person"
            for r in clean_rels
        )
        if not has_depicts:
            clean_rels.append(
                {
                    "subject": profile_acc_ids[0],
                    "predicate": "depicts_person",
                    "object": person_ids[0],
                }
            )

    # =========================================================================
    # 3. DYNAMIC & STRICT PHONE-TO-COUNTRY PARSING (ITU-T E.164 Standard)
    # =========================================================================
    if profile_acc_ids:
        main_profile = profile_acc_ids[0]
        has_country_rel = any(
            r.get("subject") == main_profile
            and r.get("predicate") == "associated_with_country"
            for r in clean_rels
        )

        if not has_country_rel:
            for r in clean_rels:
                if r.get("predicate") == "has_phone_number":
                    phone_val = str(r.get("object", "")).strip()
                    country_uri = get_country_uri_from_phone(phone_val)

                    if country_uri:
                        clean_rels.append(
                            {
                                "subject": main_profile,
                                "predicate": "associated_with_country",
                                "object": country_uri,
                            }
                        )
                        print(
                            f"[STRICT PARSE] Derived Country '{country_uri}' from Phone '{phone_val}'"
                        )
                        break

    # =========================================================================
    # 4. TARGETED ENTITY GROUNDING (STRICTLY FOR PERSON LABELS & NAMES)
    # =========================================================================
    processed_entities = []
    for ent in entities:
        subj = ent.get("subject")
        pred = str(ent.get("predicate")).strip()
        obj = str(ent.get("object")).strip()

        stype = subject_types.get(subj, "").lower()
        is_label_pred = pred.lower() in ["label", "rdfs:label"]

        # ΚΑΝΟΝΑΣ: Εκτελούμε Grounding ΑΠΟΚΛΕΙΣΤΙΚΑ αν η οντότητα αφορά Πρόσωπο (Person)
        # Αυτό αποτρέπει την αλλοίωση σε logos, emblems & ονόματα πλατφορμών!
        if is_label_pred and img_path and "person" in stype:
            grounded_obj = ground_text_to_image_pixels(obj, img_path)
            if grounded_obj != obj:
                grounded_changes[obj] = grounded_obj  # Καταγραφή διόρθωσης
            obj = grounded_obj

        if "investment_account_page" in stype and is_label_pred:
            continue

        if "profile_page" in stype and is_label_pred:
            processed_entities.append(
                {"subject": subj, "predicate": "platform", "object": obj}
            )
            continue

        if pred.lower() == "label":
            pred = "rdfs:label"

        processed_entities.append(
            {"subject": subj, "predicate": pred, "object": obj}
        )

    graph_entry["entities"] = apply_rdf_ontology_mapping(processed_entities)
    graph_entry["relations"] = clean_rels

    return graph_entry, grounded_changes


def get_country_uri_from_phone(phone_str):
    """Parses ANY international phone number string dynamically using ITU-T standards 

    and maps it to its exact DBpedia Country URI. Returns None if invalid or not found.
    """
    try:
        parsed_num = phonenumbers.parse(phone_str, None)
        if phonenumbers.is_valid_number(parsed_num):
            country_name = geocoder.country_name_for_number(parsed_num, "en")
            if country_name:
                formatted_country = country_name.replace(" ", "_")
                return f"http://dbpedia.org/resource/{formatted_country}"
    except Exception:
        pass

    return None