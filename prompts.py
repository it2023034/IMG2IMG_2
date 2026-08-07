# =========================================================================
# 1. VISUAL DESCRIPTION PROMPT (GENERIC)
# =========================================================================

def get_description_prompt():
    return """Analyse the provided image / interface screenshot and generate a precise, highly focused semantic description.

Follow these strict filtering guidelines:
1. PLATFORM & LAYOUT IDENTIFICATION: 
   - Identify the primary interface category and exact platform branding/type (e.g., messaging views, social profiles, trading dashboards).
   - Capture key layout areas, structural visual boundaries, emblems, icons, and platform identifiers.

2. TEXTUAL PRIMACY & CORE IDENTIFIERS:
   - Extract exact text strings for names, titles, organizations, domain URLs, and numeric metrics.
   - For contact numbers and dialing codes: Extract full phone sequences and international dialing prefixes.
   - For geopolitical references: Record explicit physical countries or locations.
   - For key quantitative panels: Capture account ownership, platform context, main values, and explicit positive/negative indicators.

3. EXCLUSION OF SECONDARY NOISE:
   - Exclude status updates, general post contents, feed item details, minor operational buttons, UI control elements, and temporary metadata.

Generate a clean, structured semantic description focused strictly on core entities and primary traits."""


# =========================================================================
# 2. NAMED ENTITY RECOGNITION (NER) PROMPT (GENERIC)
# =========================================================================

def get_ner_prompt(description):
    return f"""ROLE:
You are an expert Visual & Interface Named Entity Recognition (NER) system. Your task is to identify and initialize discrete entities from the input text using ONLY the allowed target ontology.

ALLOWED ONTOLOGY CLASSES (WHITELIST):
- Profile_Page, Investment_Account_Page
- Person, Location, Organisation, Visual_Symbol

EXTRACTION RULES:
1. ONTOLOGY BOUNDING & ENTITY IDS:
   - Assign entity classes strictly using ONLY the provided whitelist.
   - Every initialized entity MUST have a unique numeric ID suffix per class starting from 1 (e.g., Profile_Page_1, Person_1, Location_1, Visual_Symbol_1).

2. TARGETED DBPEDIA ENTITY LINKING (PROFILE_PAGE & LOCATION ONLY):
   - ONLY for `Profile_Page` and `Location`: You MUST format the label strictly as a canonical DBpedia resource URI (e.g., "http://dbpedia.org/resource/<Resolved_Name>").
   - FOR ALL OTHER CLASSES (`Investment_Account_Page`, `Person`, `Organisation`, `Visual_Symbol`): Format the label strictly as the raw text string extracted from the description.

3. SEMANTIC ENTITY BOUNDING:
   - Location entities MUST represent distinct geopolitical entities, sovereign states, or countries. Numeric dialing prefixes are NOT locations.
   - Profile/Account entities MUST represent the target software platform instance.

4. SEPARATION OF GRAPH NODES VS. LITERAL ATTRIBUTES:
   - Do NOT instantiate Entity IDs for literal attributes (such as roles, URLs, phone numbers, dialing codes, balances, or profits). Literals are processed strictly during relation extraction as raw text strings.

5. MANDATORY INITIALIZATION:
   - Output both `rdf:type` and `rdfs:label` for EVERY initialized entity.

OUTPUT TEMPLATE:
Class_Name_1 | rdf:type | Class_Name
Class_Name_1 | rdfs:label | "exact text string or DBpedia URI"

INPUT TEXT:
"{description}"

NER TRIPLES:
""".strip()


# =========================================================================
# 3. UNIFIED RELATION EXTRACTION (RE) PROMPT (GENERIC)
# =========================================================================

# =========================================================================
# 3. UNIFIED RELATION EXTRACTION (RE) PROMPT
# =========================================================================

def get_relation_extraction_prompt(description, schema, examples, extracted_entities):
    return f"""ROLE:
You are an expert, deterministic Knowledge Graph Relation Extraction Engine. Your task is to connect initialized entities and literal attributes into a valid graph based strictly on the allowed schema and explicit text evidence.

ALLOWED ONTOLOGY METAPATHS:
{schema}

AVAILABLE INITIALIZED ENTITIES:
{extracted_entities}

FEW-SHOT SYNTAX EXAMPLES:
{examples}

GRAPH GRAMMAR & CONNECTION RULES:

1. STRICT METAPATH DOMAIN & RANGE COMPLIANCE:
   - Construct relation triples ONLY by strictly matching valid Subject-Predicate-Object patterns declared in ALLOWED ONTOLOGY METAPATHS.
   - Never assign a predicate to an unapproved Subject class.

2. FACTUAL PHONE CODE RULE (NO LITERAL PLACEHOLDERS OR INFERENCES):
   - ONLY issue a `has_phone_code` relation if an EXPLICIT numeric dialing prefix (e.g., sequence starting with '+' or digits) is explicitly stated in the text description.
   - NEVER output data-type placeholders (e.g., "String") or infer dialing codes from Location entity labels or geographic names. If no numeric dialing prefix exists in the text, SKIP the `has_phone_code` relation entirely.

3. STRICT PHONE NUMBER LITERAL RULE:
   - The predicate `has_phone_number` MUST ONLY accept objects that are explicit numeric telephone strings (containing digits, spaces, or leading '+').
   - NEVER map proper names, entity labels, or non-numeric strings to `has_phone_number`. If no numeric telephone string is explicitly present in the description, SKIP the `has_phone_number` relation entirely.

4. MANDATORY EXHAUSTIVE CONNECTIVITY (NO ISOLATED OR OMITTED NODES):
   - YOU MUST CONNECT EVERY SINGLE ENTITY listed under AVAILABLE INITIALIZED ENTITIES.
   - It is strictly forbidden to leave any initialized Entity ID (e.g., Person, Organisation, Location, Profile_Page, Investment_Account_Page) disconnected or omitted from the graph.
   - For every Person entity present, you MUST establish their structural or associative relation to the primary container page/account (e.g., via `depicts_person` or equivalent allowed metapath) and attach all corresponding literal attributes (e.g., `has_role`).
   - Every initialized node MUST participate in at least one relation triple.
   
5. DYNAMIC LITERAL ATTACHMENT:
   - When mapping literal predicates (e.g., `has_role`, `has_url`, `has_phone_number`, `has_total_balance`, `has_total_profit`, `has_total_loss`), attach the exact extracted text value as a string literal.
   - Never output schema type names (like "String") as literal values.

6. CLEAN OUTPUT FORMAT:
   - Output ONLY valid relation triples in the exact syntax: `Subject_ID | predicate | Object_ID_or_Literal`.
   - Do NOT include markdown blocks, notes, or conversational text.

INPUT TEXT DESCRIPTION:
"{description}"

RELATION TRIPLES:
""".strip()


# =========================================================================
# 4. COUNTERFACTUAL PROMPTS (GENERIC)
# =========================================================================

def get_counterfactual_prompt(original_graph, detected_change):
    return f"""ROLE:
You are a strict Graph Editing Engine. Your job is to modify exactly ONE triple from the original knowledge graph to reflect a visually detected change.

ORIGINAL GRAPH TRIPLES:
{original_graph}

DETECTED VISUAL CHANGE:
"{detected_change}"

STRICT COMPILER RULES:
1. Analyze the DETECTED VISUAL CHANGE to identify which specific visual attribute, entity label, or text string was modified.
2. Locate the corresponding Entity ID in ORIGINAL GRAPH TRIPLES that represents that element.
3. Target the 'rdfs:label' predicate of that specific Entity ID to update its value.
4. Output EXACTLY three structured lines following the template below.

OUTPUT TEMPLATE:
TARGET_SUBJECT: <Entity_ID>
TARGET_PREDICATE: rdfs:label
NEW_OBJECT: <New_Label_Value>
""".strip()


def get_geometric_counterfactual_prompt(change_from):
    return f"""You are a strict digital interface and counterfactual asset generator.
The original interface contains an element described as: '{change_from}'.

CRITICAL CONSTRAINT: Propose ONE alternative, logically coherent text string, identity name, role, visual symbol, or emblem that serves as a direct structural replacement for '{change_from}' within the same interface context.

Respond with ONLY the name/value of the proposed replacement asset (1-4 words max). No explanations, no code tags, no conversational filler."""