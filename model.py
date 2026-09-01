import ollama
from ollama import Client

# Αρχικοποίηση του Ollama client στη τοπική θύρα 11434
client = Client(host='http://localhost:11434')

def generate_description(images, prompt):
    # Δημιουργία περιγραφής εικόνας/διεπαφής με χρήση του multimodal μοντέλου gemma3:12b.
    if isinstance(images, str):
        images = [images]
    
    flat_images = [str(img) for img in images]

    response = client.generate(
        model='gemma3:12b',
        prompt=prompt,
        images=flat_images,
        options={
            'temperature': 0.0,
            'top_p': 0.1,      
            'top_k': 1     
        }
    )
    
    return response['response'].strip()

def extract_entities(description, prompt_func):
    # Εξαγωγή οντοτήτων (NER) από τη κειμενική περιγραφή βάσει της ορισμένης οντολογίας.
    full_prompt = prompt_func(description)
    response = client.generate(
        model='gemma3:12b',
        prompt=full_prompt,
        options={
            'temperature': 0,
            'top_p': 0.1,
            'num_predict': 500
        }
    )
    return response['response']

def extract_triples(description, schema, examples, extracted_entities, prompt_func):
    # Εξαγωγή σχέσεων (triples) μεταξύ των οντοτήτων και των ιδιοτήτων τους για το Knowledge Graph.
    full_prompt = prompt_func(description, schema, examples, extracted_entities)
    response = client.generate(
        model='gemma3:12b',
        prompt=full_prompt,
        options={
            'temperature': 0,
            'top_p': 0.1,
            'num_predict': 500
        }
    )
    return response['response']