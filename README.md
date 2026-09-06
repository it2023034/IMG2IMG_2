# Automated Counterfactual Image & Knowledge Graph Generator

An end-to-end OSINT-oriented framework designed to automate the generation of counterfactual visual assets and update their corresponding Knowledge Graphs (KG) with strict geometric, visual, and semantic constraints.

This pipeline identifies key entities in visual screenshots (e.g., social media profiles, trading dashboards, messaging apps), performs sub-word bounding box isolation, generates semantically valid counterfactual text replacements, and renders them onto the target image with pixel-perfect typography matching.

---

## 🏗️ System Architecture

The pipeline consists of three core modules working in sequence:

1. Entity Extraction & Knowledge Graph Generation (EasyOCR + LLM)
   - Input: Screenshot Image
   - Outputs: description.txt & original_graph.json

2. Counterfactual Entity Selection & Strict Validation
   - Enforces: Gender Preservation, Initials Matching, Strict Length Match (±1 char)

3. Localized Inpainting & Pixel-Perfect Rendering (FLUX.2 + OpenCV)
   - Outputs: counterfactual_image.png & counterfactual_results.json

---

## 🔑 Key Features

* Sub-Word Bounding Box Isolation: Uses EasyOCR to detect multi-word strings (e.g., "Currently in Syria") and mathematically isolates only the target entity ("Syria"), leaving surrounding UI text untouched.
* Strict Geometric & Semantic Constraints:
  - Gender Preservation: Invariant female-to-female and male-to-male replacement logic.
  - Length Matching: Generates replacements within a strict ±1 character tolerance.
  - Initials & Category Alignment: Retains exact word count and matching initial letters.
* Platform-Aware Visual Fidelity:
  - Facebook UI Specialization: Applies Top-Edge Baseline Alignment, dynamic X-axis padding, and Morphological Dilation (cv2.dilate) for authentic bold typography.
  - General UI Safety Mode: Uses conservative ROI bounding for dark dashboards, messaging platforms (Viber, WhatsApp), preventing damage to adjacent icons, arrows, or badges.
* Knowledge Graph Consistency: Automatically synchronizes DBpedia URIs and RDF triples (owl:sameAs, rdfs:label) with the new counterfactual values.

---

## 🛠️ Repository Structure

* images/ - Input raw screenshots
* results/ - Generated text descriptions & original KGs (.json)
* counterfactual/ - Output modified counterfactual images
* counterfactual_results/ - Output updated Knowledge Graphs (.json)
* pipeline.py - Main pipeline CLI orchestrator
* pipeline_cf.py - Core Image Processing & FLUX Rendering Engine
* prompts.py - Structured LLM Prompts & Geometric Constraint Logic
* model.py - LLM Inference Wrapper (Ollama / Gemma)
* utils.py - Graph processing, filtering, & deduplication utilities

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.10+
* CUDA-capable GPU (Recommended: >= 16 GB VRAM)
* Ollama running locally with gemma3:12b (or preferred LLM)

### Environment Setup

1. Clone the Repository:
   git clone https://github.com/your-username/counterfactual-image-kg.git
   cd counterfactual-image-kg

2. Install Dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers transformers easyocr opencv-python numpy pillow

3. Verify Ollama Model:
   ollama pull gemma3:12b

---

## 💡 Usage

Run the main pipeline controller:

python pipeline.py

### Pipeline Execution Flow (Menu Options):

1. [1] Extract Original Triples
   - Processes input images in images/.
   - Generates scene descriptions, extracts NER entities, builds RDF triples, and saves results/<image_name>_graph.json.

2. [2] Generate Counterfactual Image (FLUX Pixels)
   - Selects an entity to replace via LLM.
   - Enforces strict geometric rules (Gender, Length ±1, Initials).
   - Unloads Ollama VRAM and loads FLUX.2-klein-9B.
   - Applies sub-pixel inpainting/blending and exports counterfactual/<image_name>_cf.png.

3. [3] Apply Graph Edit on Counterfactual (JSON Logic)
   - Synchronizes the generated counterfactual change back into the Knowledge Graph structure.
   - Saves updated triples to counterfactual_results/.

---

## ⚙️ Configuration & Fine-Tuning

### Custom UI Adjustments (pipeline_cf.py)

Platform-specific rendering is determined automatically via description matching:

* Facebook Platform Mode:
  - Scaling Factor: 0.82
  - Typography: Segoe UI / Roboto
  - Line Alignment: Top-Edge Alignment (orig_y_top_local)
  - Morphological Weight: 2x2 Ellipse Dilate (Bold effect)

* Standard UI Mode (Dashboards / Viber / General):
  - Scaling Factor: 0.72
  - Padding: Strict 0.15 * text_h (Prevents UI icon distortion)
  - Line Alignment: Vertical Center Alignment
