import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def predict(prompt: str):
    model_path = "./models/luna_style_training-10.safetensors"

    # Load your model using diffusers or another method (you might need to adjust this)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
        safety_checker=None,
        revision="fp16"
    ).to("cuda")

    # Load LoRA weights
    pipe.load_lora_weights(model_path)

    image = pipe(prompt).images[0]
    image.save("/tmp/output.png")
    return {"image": "/tmp/output.png"}
