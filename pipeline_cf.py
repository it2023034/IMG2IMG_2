import os
import gc
import cv2
import torch
import easyocr
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

# Ορισμός διαδρομής cache για το Hugging Face
os.environ["HF_HOME"] = "/var/local/storage/it2023034/hf_cache"

class CreateCounterfactualImage:
    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-9B"):
        print(f"Loading {model_id} onto GPU...")
        
        # Αρχικοποίηση Diffusion Pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )

        # Βελτιστοποίηση μνήμης GPU
        self.pipe.enable_model_cpu_offload(device=torch.device("cuda:0"))
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Αρχικοποίηση EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=True)

    def _get_text_details(self, image_pil, target_text):
        """Εντοπισμός συντεταγμένων. Απομονώνει τη λέξη-στόχο με ακρίβεια pixel."""
        img_np = np.array(image_pil)
        results = self.reader.readtext(img_np)
        
        target_clean = " ".join(target_text.lower().strip().split())
        
        for (bbox, text, prob) in results:
            found_text_clean = " ".join(text.lower().strip().split())
            
            if target_clean in found_text_clean:
                pts = np.array(bbox, dtype=np.int32)
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                
                words = found_text_clean.split()
                if len(words) > 1 and target_clean in words:
                    full_w = x_max - x_min
                    target_idx = words.index(target_clean)
                    
                    char_counts = [len(w) for w in words]
                    total_chars = sum(char_counts) + len(words) - 1
                    
                    start_char = sum(char_counts[:target_idx]) + target_idx
                    end_char = start_char + len(target_clean)
                    
                    sub_x1 = max(x_min, x_min + int(full_w * (start_char / total_chars)) - 2)
                    sub_x2 = min(x_max, x_min + int(full_w * (end_char / total_chars)) + 2)
                    
                    sub_pts = np.array([
                        [sub_x1, y_min],
                        [sub_x2, y_min],
                        [sub_x2, y_max],
                        [sub_x1, y_max]
                    ], dtype=np.int32)
                    
                    return (int(sub_x1), int(y_min), int(sub_x2), int(y_max)), sub_pts, True
                
                return (int(x_min), int(y_min), int(x_max), int(y_max)), pts, False
        
        return None, None, False

    def generate(self, change_from: str, change_to: str, img_path: str, font_scale: float = None):
        """Εκτέλεση αντικατάστασης κειμένου με διαχωρισμένη λογική (Strict IF-ELSE for Facebook)."""
        full_image = load_image(img_path).convert("RGB")
        full_image_np = np.array(full_image)
        width, height = full_image.size
        
        clean_from = " ".join(change_from.strip().split())
        clean_to = " ".join(change_to.strip().split())

        # --- ΕΛΕΓΧΟΣ ΑΝ ΠΡΟΚΕΙΤΑΙ ΓΙΑ FACEBOOK UI ---
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        desc_path = os.path.join("results", f"{base_name}_desc.txt")
        is_facebook = False
        
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                desc_content = f.read().lower()
            if "facebook" in desc_content:
                is_facebook = True
        
        bbox, poly_pts, is_subword = self._get_text_details(full_image, clean_from)
        
        if bbox is None or poly_pts is None:
            print(f"Warning: Text '{clean_from}' not found.")
            return full_image

        orig_x1, orig_y1, orig_x2, orig_y2 = bbox
        text_h = orig_y2 - orig_y1
        orig_w = orig_x2 - orig_x1
        
        # ==========================================
        # 1. PADDING & ROI SELECTION
        # ==========================================
        if is_facebook:
            # Facebook specific: Μεγαλύτερο δεξί padding για subwords (e.g. Currently in Syria -> Serbia)
            len_ratio = max(1.0, len(clean_to) / max(1, len(clean_from)))
            pad_x_left = int(text_h * 0.3)
            pad_x_right = int(text_h * (2.2 * len_ratio))
            pad_y = int(text_h * 0.6)
        else:
            # General UI (Dashboard, Viber, etc.): Αυστηρά μικρό padding για προστασία γειτονικών icons/arrows
            pad_x_left = int(text_h * 0.15)
            pad_x_right = int(text_h * 0.3)
            pad_y = int(text_h * 0.3)
        
        x1 = max(0, orig_x1 - pad_x_left)
        y1 = max(0, orig_y1 - pad_y)
        x2 = min(width, orig_x2 + pad_x_right)
        y2 = min(height, orig_y2 + pad_y)
        
        crop_np = full_image_np[y1:y2, x1:x2].copy()
        
        poly_pts_local = poly_pts - [x1, y1]
        orig_text_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(orig_text_mask, [poly_pts_local], 255)

        # ==========================================
        # 2. INPAINTING & CLEANING
        # ==========================================
        if is_facebook:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean_mask = cv2.dilate(orig_text_mask, kernel, iterations=2)
            cleaned_crop_np = cv2.inpaint(crop_np, clean_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            clean_mask = cv2.dilate(orig_text_mask, kernel, iterations=1)
            cleaned_crop_np = cv2.inpaint(crop_np, clean_mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
        
        bg_color = np.median(cleaned_crop_np, axis=(0, 1)).astype(np.uint8)
        brightness = np.mean(bg_color)
        
        color_prompt = "solid bright white UI text" if brightness < 128 else "solid dark charcoal grey UI text"

        input_crop_np = np.zeros_like(cleaned_crop_np)
        input_crop_np[:, :] = bg_color
        
        high_res_input = Image.fromarray(input_crop_np).resize((1024, 256), Image.Resampling.LANCZOS)
        
        # ==========================================
        # 3. PROMPT GENERATION
        # ==========================================
        if is_facebook:
            prompt_str = f"{color_prompt} reading exactly '{clean_to}' in clean Segoe UI Roboto system font, wide kerning, natural letter spacing, sharp regular typography, flat background."
        else:
            prompt_str = f"{color_prompt} reading exactly '{clean_to}' in clean regular sans-serif font, sharp typography, flat background."
        
        generator = torch.Generator("cuda").manual_seed(42)
                    
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.pipe(
                    prompt=prompt_str,
                    image=high_res_input,
                    num_inference_steps=24,
                    guidance_scale=2.5,
                    generator=generator
                )
        
        high_res_gen = result.images[0]
        gen_crop = high_res_gen.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
        gen_crop_np = np.array(gen_crop)
        
        diff = cv2.absdiff(gen_crop_np, input_crop_np)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        _, raw_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
        
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_clean)

        # ==========================================
        # 4. SCALING & RESIZING
        # ==========================================
        y_indices, x_indices = np.where(raw_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            gen_text_h = np.max(y_indices) - np.min(y_indices)
            gen_text_w = np.max(x_indices) - np.min(x_indices)
            
            if font_scale is not None:
                scale_factor = font_scale
            else:
                scale_factor = 0.82 if is_facebook else 0.72
            
            scale_y = (float(text_h) / float(gen_text_h)) * scale_factor if gen_text_h > 0 else 1.0
            scale_x = scale_y 

            h_m, w_m = raw_mask.shape
            new_w = max(1, int(w_m * scale_x))
            new_h = max(1, int(h_m * scale_y))
            
            scaled_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_gen = cv2.resize(gen_crop_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if is_facebook:
                # Παχαίνουμε τη μάσκα κειμένου κατά 1px για να δημιουργήσουμε τέλειο bold εφέ
                bold_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                scaled_mask = cv2.dilate(scaled_mask, bold_kernel, iterations=1)
            elif brightness <= 200:
                sharpen_kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]], dtype=np.float32)
                scaled_gen = cv2.filter2D(scaled_gen, -1, sharpen_kernel)
            
            canvas_w = max(w_m, new_w)
            text_mask = np.zeros((h_m, canvas_w), dtype=np.uint8)
            gen_crop_np_scaled = np.zeros((h_m, canvas_w, 3), dtype=np.uint8)
            
            slice_h, slice_w = min(new_h, h_m), min(new_w, canvas_w)
            text_mask[:slice_h, :slice_w] = scaled_mask[:slice_h, :slice_w]
            gen_crop_np_scaled[:slice_h, :slice_w] = scaled_gen[:slice_h, :slice_w]
            gen_crop_np = gen_crop_np_scaled
            
            if canvas_w > cleaned_crop_np.shape[1]:
                pad_w = canvas_w - cleaned_crop_np.shape[1]
                cleaned_crop_np = np.pad(cleaned_crop_np, ((0, 0), (0, pad_w), (0, 0)), mode='edge')

        # ==========================================
        # 5. ALIGNMENT & AFFINE TRANSFORMATION
        # ==========================================
        y_indices, x_indices = np.where(text_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            gen_x1 = np.min(x_indices)
            gen_x2 = np.max(x_indices)
            gen_x_center = (gen_x1 + gen_x2) / 2.0
            
            orig_x1_local = orig_x1 - x1
            orig_x2_local = orig_x2 - x1
            orig_x_center_local = (orig_x1_local + orig_x2_local) / 2.0
            
            crop_width = x2 - x1
            
            is_centered = abs(orig_x_center_local - (crop_width / 2.0)) < (crop_width * 0.15)
            is_right_aligned = (crop_width - orig_x2_local) < (crop_width * 0.1)
            
            if is_centered:
                shift_x = int(orig_x_center_local - gen_x_center)
            elif is_right_aligned:
                shift_x = int(orig_x2_local - gen_x2)
            else:
                shift_x = int(orig_x1_local - gen_x1)
                
            # Διαφορετική Y στοίχιση για Facebook (Top-line baseline) vs General (Center)
            if is_facebook:
                gen_y_top = np.min(y_indices)
                orig_y_top_local = orig_y1 - y1
                shift_y = int(orig_y_top_local - gen_y_top)
                if is_subword:
                    shift_y -= 2
            else:
                gen_y_center = (np.min(y_indices) + np.max(y_indices)) / 2.0
                orig_y_center_local = (orig_y1 + orig_y2) / 2.0 - y1
                shift_y = int(orig_y_center_local - gen_y_center)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            text_mask = cv2.warpAffine(text_mask, M, (text_mask.shape[1], text_mask.shape[0]))
            gen_crop_np = cv2.warpAffine(gen_crop_np, M, (gen_crop_np.shape[1], gen_crop_np.shape[0]))

        # ==========================================
        # 6. BLENDING & FINAL COMPOSITING
        # ==========================================
        k_size = 3
        alpha_blur = cv2.GaussianBlur(text_mask, (k_size, k_size), 0) / 255.0
        alpha_3d = np.stack([alpha_blur] * 3, axis=-1)
        
        final_crop_np = (gen_crop_np * alpha_3d + cleaned_crop_np * (1.0 - alpha_3d)).astype(np.uint8)
        final_crop_np = final_crop_np[:y2 - y1, :x2 - x1]

        final_image_np = full_image_np.copy()
        final_image_np[y1:y2, x1:x2] = final_crop_np
        
        final_image = Image.fromarray(final_image_np)

        del result
        gc.collect()
        torch.cuda.empty_cache()
        return final_image