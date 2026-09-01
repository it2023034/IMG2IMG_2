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
        """Εντοπισμός συντεταγμένων και πολυγώνου του κειμένου-στόχου."""
        img_np = np.array(image_pil)
        results = self.reader.readtext(img_np)
        
        target_clean = " ".join(target_text.lower().strip().split())
        
        for (bbox, text, prob) in results:
            found_text_clean = " ".join(text.lower().strip().split())
            if target_clean in found_text_clean or found_text_clean in target_clean:
                pts = np.array(bbox, dtype=np.int32)
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                return (int(x_min), int(y_min), int(x_max), int(y_max)), pts
        
        return None, None

    def generate(self, change_from: str, change_to: str, img_path: str, font_scale: float = None):
        """Εκτέλεση αντικατάστασης κειμένου με αυτόματη στοίχιση και κλιμάκωση."""
        full_image = load_image(img_path).convert("RGB")
        full_image_np = np.array(full_image)
        width, height = full_image.size
        
        clean_from = " ".join(change_from.strip().split())
        clean_to = " ".join(change_to.strip().split())
        
        # Εντοπισμός ορίων αρχικού κειμένου
        bbox, poly_pts = self._get_text_details(full_image, clean_from)
        
        if bbox is None or poly_pts is None:
            print(f"Warning: Text '{clean_from}' not found.")
            return full_image

        orig_x1, orig_y1, orig_x2, orig_y2 = bbox
        text_h = orig_y2 - orig_y1
        
        # Υπολογισμός περιοχής αποκοπής (ROI) με οριζόντιο/κατακόρυφο padding
        pad_x_left = int(text_h * 0.4)
        pad_x_right = int(text_h * 3.5) 
        pad_y = int(text_h * 0.8)
        
        x1 = max(0, orig_x1 - pad_x_left)
        y1 = max(0, orig_y1 - pad_y)
        x2 = min(width, orig_x2 + pad_x_right)
        y2 = min(height, orig_y2 + pad_y)
        
        crop_np = full_image_np[y1:y2, x1:x2].copy()
        
        # Αφαίρεση παλιού κειμένου με Inpainting
        poly_pts_local = poly_pts - [x1, y1]
        clean_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(clean_mask, [poly_pts_local], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)
        
        cleaned_crop_np = cv2.inpaint(crop_np, clean_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        
        # Προσδιορισμός χρώματος γραμματοσειράς βάσει φωτεινότητας φόντου
        bg_color = np.median(cleaned_crop_np, axis=(0, 1)).astype(np.uint8)
        brightness = np.mean(bg_color)
        
        color_prompt = "solid bright white UI text" if brightness < 128 else "solid dark grey UI text"

        input_crop_np = np.zeros_like(cleaned_crop_np)
        input_crop_np[:, :] = bg_color
        
        high_res_input = Image.fromarray(input_crop_np).resize((1024, 256), Image.Resampling.LANCZOS)
        
        # Παραγωγή νέου κειμένου μέσω FLUX
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
        
        # Απομόνωση μάσκας νέου κειμένου & προσαρμογή μεγέθους
        diff = cv2.absdiff(gen_crop_np, input_crop_np)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        _, raw_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_clean)
        
        y_indices, x_indices = np.where(raw_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            gen_text_h = np.max(y_indices) - np.min(y_indices)
            
            scale_factor = font_scale if font_scale is not None else 0.82
            
            scale_y = (float(text_h) / float(gen_text_h)) * scale_factor if gen_text_h > 0 else 1.0
            scale_x = scale_y 
            
            h_m, w_m = raw_mask.shape
            new_w = max(1, int(w_m * scale_x))
            new_h = max(1, int(h_m * scale_y))
            
            scaled_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_gen = cv2.resize(gen_crop_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Δημιουργία καμβά προσαρμοσμένου στο νέο πλάτος
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

        # Αυτόματη στοίχιση (Left, Center, Right) και μετατόπιση
        y_indices, x_indices = np.where(text_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            gen_x1 = np.min(x_indices)
            gen_x2 = np.max(x_indices)
            gen_x_center = (gen_x1 + gen_x2) / 2.0
            gen_y_center = (np.min(y_indices) + np.max(y_indices)) / 2.0
            
            orig_x1_local = orig_x1 - x1
            orig_x2_local = orig_x2 - x1
            orig_x_center_local = (orig_x1_local + orig_x2_local) / 2.0
            orig_y_center_local = (orig_y1 + orig_y2) / 2.0 - y1
            
            crop_width = x2 - x1
            
            is_centered = abs(orig_x_center_local - (crop_width / 2.0)) < (crop_width * 0.15)
            is_right_aligned = (crop_width - orig_x2_local) < (crop_width * 0.1)
            
            if is_centered:
                shift_x = int(orig_x_center_local - gen_x_center)
            elif is_right_aligned:
                shift_x = int(orig_x2_local - gen_x2)
            else:
                shift_x = int(orig_x1_local - gen_x1)
                
            shift_y = int(orig_y_center_local - gen_y_center)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            text_mask = cv2.warpAffine(text_mask, M, (text_mask.shape[1], text_mask.shape[0]))
            gen_crop_np = cv2.warpAffine(gen_crop_np, M, (gen_crop_np.shape[1], gen_crop_np.shape[0]))

        # Ομαλή ανάμειξη (Alpha Blending) με το υπόβαθρο
        alpha_blur = cv2.GaussianBlur(text_mask, (3, 3), 0) / 255.0
        alpha_3d = np.stack([alpha_blur] * 3, axis=-1)
        
        final_crop_np = (gen_crop_np * alpha_3d + cleaned_crop_np * (1.0 - alpha_3d)).astype(np.uint8)
        
        final_crop_np = final_crop_np[:y2 - y1, :x2 - x1]

        # Ενσωμάτωση του αποτελέσματος στην αρχική εικόνα
        final_image_np = full_image_np.copy()
        final_image_np[y1:y2, x1:x2] = final_crop_np
        
        final_image = Image.fromarray(final_image_np)

        # Καθαρισμός μνήμης GPU
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return final_image