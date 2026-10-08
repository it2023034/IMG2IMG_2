from ollama import Client

# Initialize Ollama client pointing to the local instance
client = Client(host="http://localhost:11434")

DEFAULT_MODEL = "gemma3:12b"

def generate_description(images, prompt, model_name=DEFAULT_MODEL):
    """Generates an image/UI description using the specified multimodal model."""
    if isinstance(images, str):
        images = [images]

    # Ensure all image paths/inputs are formatted as strings
    flat_images = [str(img) for img in images]

    response = client.generate(
        model=model_name,
        prompt=prompt,
        images=flat_images,
        options={
            "temperature": 0.0,  # Zero temperature for deterministic output
            "top_p": 0.1,
            "top_k": 1,
        },
    )

    return response["response"].strip()


def extract_entities(description, prompt_func, model_name=DEFAULT_MODEL):
    """Extracts Named Entities (NER) from the text description based on a defined ontology."""
    full_prompt = prompt_func(description)
    response = client.generate(
        model=model_name,
        prompt=full_prompt,
        options={"temperature": 0, "top_p": 0.1, "num_predict": 500},
    )
    return response["response"]


def extract_triples(
    description,
    schema,
    examples,
    extracted_entities,
    prompt_func,
    model_name=DEFAULT_MODEL,
):
    """Extracts Knowledge Graph triples (subject, predicate, object) from the description."""
    full_prompt = prompt_func(
        description, schema, examples, extracted_entities
    )
    response = client.generate(
        model=model_name,
        prompt=full_prompt,
        options={"temperature": 0, "top_p": 0.1, "num_predict": 500},
    )
    return response["response"]