import os
import gc
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# Ορίζουμε το cache του HuggingFace στον τοπικό γρήγορο δίσκο
os.environ["HF_HOME"] = "/var/local/storage/it2023034/hf_cache"

class CreateCounterfactualImage:
    def __init__(self):
        print("Loading FLUX.2-klein-9B onto allocated GPU...")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.float16
        )

        # Το Slurm δίνει αυτόματα τη διαθέσιμη GPU ως cuda:0
        self.pipe.enable_model_cpu_offload(device=torch.device("cuda:0"))
        
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate(self, change_from, change_to, img_path):
        input_image = load_image(img_path)
        prompt_str = f"""
            In this interface layout, modify the specific element or identity described as '{change_from}' and visually replace it with '{change_to}'.
            Keep the rest image exactly as it is
            Dont add or replace or remove anything else
            composition, clothing style, pose, objects, lightning and scene details. 
            Do not modify any other part of the image.
        """
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                result = self.pipe(
                    image=input_image,
                    prompt=prompt_str,
                    num_inference_steps=4,
                    guidance_scale=3,
                )
        image = result.images[0]
        
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return image