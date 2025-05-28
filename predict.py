import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class Predictor(BasePredictor):
    def setup(self):
        # Load your trained LoRA model here
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

        # Load the LoRA weights
        self.pipe.load_lora_weights("./models/luna_style_training-10.safetensors")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
    ) -> Path:
        image = self.pipe(prompt).images[0]
        out_path = "/tmp/output.png"
        image.save(out_path)
        return Path(out_path)

