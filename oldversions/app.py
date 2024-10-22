import os
import time
import torch
from pathlib import Path
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# Configuration Variables
CONFIG = {
    "src_dir": "./src",  # Source directory for images
    "dst_dir": "./dst",  # Destination directory for output images
    "checkpoint": "lllyasviel/control_v11p_sd15_softedge",  # ControlNet checkpoint
    "prompt": "A 17th century painting, family, group, holding hands, outdoors",  # Text prompt for the diffusion model
    "num_inference_steps": 40,  # Number of inference steps for diffusion
    "controlnet_strength": 0.8,  # Strength of the controlnet effect (range: 0-1)
    "guidance_scale": 5,  # Higher values encourage closer adherence to the prompt
    "image_resolution": (512, 512),  # Resolution of generated images
    "use_safe_mode": True,  # Whether to use the safe mode for the processor
    "seed": 123456789,  # Seed for reproducibility
}

# Directories
SRC_DIR = Path(CONFIG["src_dir"])
DST_DIR = Path(CONFIG["dst_dir"])
CHECKPOINT = CONFIG["checkpoint"]
PROMPT = CONFIG["prompt"]

# Ensure the directories exist
SRC_DIR.mkdir(parents=True, exist_ok=True)
DST_DIR.mkdir(parents=True, exist_ok=True)

# Load model and processor
print("Loading models...")
processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
controlnet = ControlNetModel.from_pretrained(CHECKPOINT, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.manual_seed(CONFIG["seed"])

def process_image(image_path):
    print(f"Processing image: {image_path}")
    
    # Load and process input image
    image = load_image(image_path).resize(CONFIG["image_resolution"])
    control_image = processor(image, safe=CONFIG["use_safe_mode"])

    # Create output filename
    output_image_path = DST_DIR / (Path(image_path).stem + "_out.png")
    
    # Run diffusion model
    generated_image = pipe(
        PROMPT, 
        num_inference_steps=CONFIG["num_inference_steps"], 
        generator=generator, 
        image=control_image, 
        controlnet_conditioning_scale=CONFIG["controlnet_strength"],
        guidance_scale=CONFIG["guidance_scale"]
    ).images[0]

    # Save the generated image
    generated_image.save(output_image_path)
    print(f"Saved output image: {output_image_path}")

class ImageEventHandler(FileSystemEventHandler):
    """Event handler for new image files in the src directory."""
    
    def on_created(self, event):
        # Check if the new file is an image
        if event.is_directory:
            return
        
        file_extension = Path(event.src_path).suffix.lower()
        if file_extension in ['.png', '.jpg', '.jpeg']:
            process_image(event.src_path)

if __name__ == "__main__":
    # Set up watchdog observer
    event_handler = ImageEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=SRC_DIR, recursive=False)
    
    # Start the observer
    observer.start()
    print(f"Watching {SRC_DIR} for new images...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
