def get_description_prompt():
    """Generates a prompt for creating a precise, highly focused semantic description

    of an image or user interface screenshot.
    """
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
    """Generates a prompt for Named Entity Recognition (NER) initialization based on a target ontology."""
    return f"""ROLE:
You are an expert Visual & Interface Named Entity Recognition (NER) system. Your task is to identify and initialize discrete entities from the input text using ONLY the allowed target ontology.

ALLOWED ONTOLOGY CLASSES (WHITELIST):
- Profile_Page, Investment_Account_Page
- Person, Organisation, Visual_Symbol

EXTRACTION RULES:
1. ONTOLOGY BOUNDING & STRICT SEQUENTIAL IDS:
- Assign entity classes strictly using ONLY the provided whitelist.
- Every initialized entity MUST have a unique numeric ID suffix per class, strictly starting sequentially from 1 (e.g., Profile_Page_1, Investment_Account_Page_1, Person_1, Organisation_1, Visual_Symbol_1).

2. CATEGORY-BASED MANDATORY ENTITY INITIALIZATION:
- MESSAGING & CONTACT INTERFACES: For any messaging app, contact info screen, or communication profile, YOU MUST MANDATORILY INITIALIZE BOTH `Profile_Page_1` AND `Person_1`.
- SOCIAL MEDIA PROFILES: For social networking profiles, initialize `Profile_Page_1` AND `Person_1`.
- FINANCIAL & TRADING DASHBOARDS: For financial, broker, or trading dashboards, initialize `Investment_Account_Page_1` AND `Person_1`.
- ORGANISATIONS & BRANDS: If an enterprise, broker, official institution, or military entity is present, initialize `Organisation_1`.
- VISUAL EMBLEMS: If official logos or emblems are identified, initialize `Visual_Symbol` entities starting at ID 1.

3. TARGETED DBPEDIA ENTITY LINKING (PROFILE_PAGE ONLY):
- ONLY for `Profile_Page`: Format the label strictly as the canonical DBpedia resource URI of the underlying application platform (e.g., "http://dbpedia.org/resource/<Platform_Name>").
- FOR ALL OTHER CLASSES (`Investment_Account_Page`, `Person`, `Organisation`, `Visual_Symbol`): Format the label strictly as the raw text string extracted from the description.

4. SEPARATION OF GRAPH NODES VS. LITERALS & URIS:
- Do NOT instantiate Entity IDs for locations, roles, URLs, phone numbers, balances, or profits. Geographic locations are mapped directly as DBpedia URIs during relation extraction.

5. MANDATORY INITIALIZATION:
- Output both `rdf:type` and `rdfs:label` for EVERY initialized entity.

OUTPUT TEMPLATE:
<Class_Name>_<ID> | rdf:type | <Class_Name>
<Class_Name>_<ID> | rdfs:label | "<exact_text_string_or_DBpedia_URI>"

INPUT TEXT:
"{description}"

NER TRIPLES:
""".strip()


def get_relation_extraction_prompt(
    description, schema, examples, extracted_entities
):
    """Generates a prompt for Knowledge Graph relation extraction and connectivity rules."""
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

2. DIRECT DBPEDIA LOCATION & COUNTRY MAPPING RULE:
- CURRENT LOCATION (located_in): Connect Person entities to DBpedia Location URIs using `located_in` whenever the text describes current physical presence or active location (e.g., Person_<ID> | located_in | http://dbpedia.org/resource/<Location_Name>).
- ORIGIN / HOMELAND (originates_from): Connect Person entities to DBpedia Location URIs using `originates_from` whenever the text describes origin, birthplace, or homeland.
- PROFILE COUNTRY (associated_with_country): Connect Profile_Page entities directly to DBpedia Location URIs using `associated_with_country`.
- MANDATORY PHONE COUNTRY DERIVATION: Whenever a phone number containing an international dialing prefix or country indicator is present, YOU MUST ALWAYS establish the `associated_with_country` relation connecting `Profile_Page_1` directly to the DBpedia Country URI.
- NEVER create intermediate Location entity IDs. Map the target location strictly as a DBpedia URI in the Object position.

3. EXHAUSTIVE LITERAL & ATTRIBUTE ATTACHMENT:
- Extract and attach ALL explicit literal attributes present in the description to their valid subjects:
- Phone numbers via `has_phone_number`
- Financial metrics (balances, profits) via `has_total_balance` and `has_profit`
- URLs via `has_url`
- Job titles, ranks, or investor status via `has_role`
- Visual marks/emblems via `displays_symbol`

4. ROLE VS. AFFILIATION SEPARATION:
- Connect `affiliated_with` strictly from a Person to an initialized Organisation entity ID representing an official institution/company.
- Attach `has_role` strictly as a string literal representing job titles, occupations, or ranks.

5. MANDATORY EXHAUSTIVE CONNECTIVITY:
- YOU MUST CONNECT EVERY SINGLE ENTITY listed under AVAILABLE INITIALIZED ENTITIES.
- It is strictly forbidden to leave any initialized Entity ID disconnected or omitted from the graph.

6. CLEAN OUTPUT FORMAT:
- Output ONLY valid relation triples in the exact syntax: `Subject_ID | predicate | Object_ID_or_URI_or_Literal`.
- Do NOT include markdown blocks, notes, or conversational text.

INPUT TEXT DESCRIPTION:
"{description}"

RELATION TRIPLES:
""".strip()

def get_geometric_counterfactual_prompt(change_from):
    """Generates a prompt for producing string replacements with strict spatial and character length constraints."""
    char_count = len(change_from)
    words = [w for w in change_from.split() if w]
    word_count = len(words)

    initials = [w[0].upper() for w in words]
    initials_str = ", ".join(
        [
            f"Word {i+1} MUST start with '{initials[i]}'"
            for i in range(len(initials))
        ]
    )

    return f"""ROLE:
You are an expert OSINT and Knowledge Graph Counterfactual Generation Engine.

TARGET STRING TO MODIFY:
'{change_from}' (Length: {char_count} chars | Word Count: {word_count})

ANALYSIS & TASK:
1. Identify the semantic category of '{change_from}' (Person Name, Location, Role, Organisation, etc.).
2. If it is a Person Name, infer the grammatical gender (Female or Male).
3. Generate EXACTLY ONE real-world replacement string that matches the detected category, gender, and structural constraints.

STRICT CONSTRAINT RULES:
1. SEMANTIC & GENDER PRESERVATION (CRITICAL):
   - Female Person Name -> MUST generate a valid Female Person Name (NEVER male!).
   - Male Person Name -> MUST generate a valid Male Person Name (NEVER female!).
   - Location -> MUST generate a real-world Location/Country (NEVER a person name!).
   - Role/Organisation -> MUST generate a valid matching entity.

2. STRUCTURE & INITIALS MATCH (MANDATORY):
   - Exact word count required: {word_count} word(s).
   - {initials_str}.

3. STRICT CHARACTER LENGTH MATCH (CRITICAL FOR UI LAYOUT):
   - Target length: EXACTLY {char_count} characters (ideal).
   - Absolute allowed range: strictly between {max(1, char_count - 1)} and {char_count + 1} characters (MAX +-1 char).
   - The generated name MUST be a real, syntactically and orthographically correct name within this exact length constraint.

OUTPUT FORMAT:
Output ONLY the raw replacement string. Do not include quotes, markdown formatting, prefixes, or explanations.
""".strip()


def get_counterfactual_prompt(original_graph, detected_change):
    """Generates a prompt for updating Knowledge Graph triples following a visual edit."""
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