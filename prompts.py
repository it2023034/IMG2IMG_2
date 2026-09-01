def get_description_prompt():
    # Prompt για τη δημιουργία ακριβούς και εστιασμένης σημασιολογικής περιγραφής εικόνας ή διεπαφής.
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

def get_ner_prompt(description):
    # Prompt για την αναγνώριση και αρχικοποίηση οντοτήτων (NER) με βάση καθορισμένη οντολογία.
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


def get_relation_extraction_prompt(description, schema, examples, extracted_entities):
    # Prompt για την εξαγωγή και σύνδεση σχέσεων μεταξύ των αναγνωρισμένων οντοτήτων στο Knowledge Graph.
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

def get_geometric_counterfactual_prompt(change_from):
    # Prompt για τη δημιουργία εναλλακτικής τιμής (counterfactual) με βάση αυστηρούς γεωμετρικούς/ορθογραφικούς περιορισμούς.
    char_count = len(change_from)
    words = change_from.split()
    
    initials_str = " ".join([f"'{w[0].upper()}...'" for w in words]) if words else ""
    
    return f"""ROLE:
You are a strict OSINT and Knowledge Graph Counterfactual Engine.

TARGET ASSET TO REPLACE:
'{change_from}' (Length: {char_count} chars)

TASK:
Propose EXACTLY ONE real-world, natural replacement of the EXACT SAME category (Person Name, Location, Job Title, or Organisation).

STRICT RULES:
1. GENDER ALIGNMENT: If the target is a Person Name, the replacement MUST preserve the exact same gender (Female -> Female, Male -> Male).
2. INITIALS MATCH: If the target consists of multiple words, each word in your proposed replacement MUST start with the exact same initial letter ({initials_str}).
3. LENGTH MATCH: Target length is ~{char_count} characters. Keep the replacement length as close to {char_count} characters as possible (tolerance +/- 2 chars).
4. REAL-WORLD ENTITY: Output MUST be a real, meaningful entity (never random characters).

FEW-SHOT EXAMPLES:
- Target: 'Mary Jane' (Female, M J) -> Output: Markella Jane (Female)
- Target: 'John Smith' (Male, J S) -> Output: Jack Slater (Male)
- Target: 'Data Analyst' (Role, D A) -> Output: Data Architect (Role)
- Target: 'London' (Location, L) -> Output: Lisbon (Location)

FORMAT:
Output ONLY the raw string value. No quotes, no explanations, no markdown.
""".strip()


def get_counterfactual_prompt(original_graph, detected_change):
    # Prompt για την ενημέρωση του Knowledge Graph μετά από μια οπτική αλλαγή (counterfactual edit).
    return f"""ROLE:
You are a strict Graph Editing Engine for OSINT Knowledge Graphs. Your goal is to update the original knowledge graph based on a single visual counterfactual edit.

ORIGINAL GRAPH TRIPLES:
{original_graph}

DETECTED VISUAL CHANGE:
"{detected_change}"

INSTRUCTIONS:
1. Identify which entity and predicate in the original graph correspond to the detected change.
2. Select the correct predicate from the graph schema based on the type of change (e.g., 'has_role' for occupations, 'located_in' or 'originates_from' for locations, 'rdfs:label' or 'label' for names/entities).
3. Update ONLY the target triple to reflect the new value.

OUTPUT TEMPLATE (EXACTLY 3 LINES):
TARGET_SUBJECT: <Entity_ID>
TARGET_PREDICATE: <Target_Predicate>
NEW_OBJECT: <New_Value>
""".strip()