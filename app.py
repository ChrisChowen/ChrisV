import os
import time
import torch
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
import shutil
import queue
 
# Configuration Variables
CONFIG = {
    "src_dir": "./src",  # Source directory for images
    "dst_dir": "./dst",  # Destination directory for output images
    "html_dir": "./html",  # Directory for HTML and images
    "checkpoint": "lllyasviel/control_v11p_sd15_softedge",  # ControlNet checkpoint
    "prompt_base": "",  # Fixed part of the prompt
    "dynamic_prompt": "Classic oil painting, beautiful, artistic symbolism, brushstrokes, traditional medium, hand-painted, dark, gothic, impressionistic style, high-quality canvas, limited edition, oil paint, palette knife, windsor & newton oil paints, award winning",  # Placeholder for dynamic part of the prompt
    "num_inference_steps": 40,  # Number of inference steps for diffusion
    "controlnet_strength": 0.8,  # Strength of the controlnet effect (range: 0-1)
    "guidance_scale": 5,  # Higher values encourage closer adherence to the prompt
    "image_resolution": (512, 512),  # Resolution of generated images
    "use_safe_mode": True,  # Whether to use the safe mode for the processor
    "seed": 123456789,  # Seed for reproducibility
    "http_port": 8000,  # Port for HTTP server
}

# Directories
SRC_DIR = Path(CONFIG["src_dir"]).resolve()
DST_DIR = Path(CONFIG["dst_dir"]).resolve()
HTML_DIR = Path(CONFIG["html_dir"]).resolve()
IMG_DIR = HTML_DIR / "imgs"
CHECKPOINT = CONFIG["checkpoint"]

# Ensure the directories exist
SRC_DIR.mkdir(parents=True, exist_ok=True)
DST_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

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

# Global prompt (base + dynamic part)
global_prompt = f"{CONFIG['prompt_base']}, {CONFIG['dynamic_prompt']}"

def process_image(image_path):
    print(f"Processing image: {image_path}")
    
    # Copy the source image to img1.png immediately
    shutil.copy(image_path, IMG_DIR / "img1.png")
    print(f"Updated imgs/img1.png with the new source image.")

    # Temporarily replace img2.png with img3.png
    img2_path = IMG_DIR / "img2.png"
    img3_path = IMG_DIR / "img3.png"
    
    if img2_path.exists():
        # Backup img2.png by swapping with img3.png
        shutil.copy(img3_path, img2_path)
        print(f"Temporarily replaced img2.png with img3.png.")

    # Load and process input image
    image = load_image(str(image_path)).resize(CONFIG["image_resolution"])
    control_image = processor(image, safe=CONFIG["use_safe_mode"])

    # Create output filename
    output_image_path = DST_DIR / (Path(image_path).stem + "_out.png")
    
    # Run diffusion model with intermediate step callback
    def save_intermediate_steps(step, timestep, latents):
        """Save the current step's image at every 5th step."""
        if step % 5 == 0:  # Save every 5 steps
            intermediate_image = pipe.decode_latents(latents)
            intermediate_image = pipe.numpy_to_pil(intermediate_image)[0]
            # Replace img2.png with the current step's image
            intermediate_image.save(IMG_DIR / "img2.png")
            print(f"Updated imgs/img2.png with image from step {step}.")

    generated_image = pipe(
        global_prompt,  # Use the global prompt here
        num_inference_steps=CONFIG["num_inference_steps"], 
        generator=generator, 
        image=control_image, 
        controlnet_conditioning_scale=CONFIG["controlnet_strength"],
        guidance_scale=CONFIG["guidance_scale"],
        callback=save_intermediate_steps,  # Pass the callback to save images during inference
        callback_steps=5  # Set callback_steps to 5 to match saving every 5 steps
    ).images[0]

    # Save the final generated image
    generated_image.save(output_image_path)
    print(f"Saved output image: {output_image_path}")

    # Finally replace img2.png with the final generated image
    shutil.copy(output_image_path, IMG_DIR / "img2.png")
    print(f"Updated imgs/img2.png with the final generated image.")



class ImageEventHandler(FileSystemEventHandler):
    """Event handler for new image files in the src directory."""
    
    def on_created(self, event):
        # Check if the new file is an image
        if event.is_directory:
            return
        
        file_extension = Path(event.src_path).suffix.lower()
        if file_extension in ['.png', '.jpg', '.jpeg']:
            process_image(Path(event.src_path))

# Silence HTTP server logs
class SilentHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Override to silence the log messages

def run_http_server():
    os.chdir(HTML_DIR)
    handler = SilentHTTPRequestHandler
    httpd = HTTPServer(("", CONFIG["http_port"]), handler)
    print(f"Serving HTTP on port {CONFIG['http_port']}...")
    httpd.serve_forever()

# Thread to listen for prompt changes from terminal
def listen_for_prompt_changes(prompt_queue):
    while True:
        user_input = input("Enter a new prompt: ").strip()
        prompt_queue.put(user_input)

if __name__ == "__main__":
    # Start the HTTP server in a separate thread
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
    # Queue to hold new prompts
    prompt_queue = queue.Queue()

    # Start prompt listener in a separate thread
    prompt_listener_thread = threading.Thread(target=listen_for_prompt_changes, args=(prompt_queue,), daemon=True)
    prompt_listener_thread.start()

    # Set up watchdog observer
    event_handler = ImageEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(SRC_DIR), recursive=False)
    
    # Start the observer
    observer.start()
    print(f"Watching {SRC_DIR} for new images...")

    try:
        while True:
            # Check for new prompt in the queue
            if not prompt_queue.empty():
                new_prompt = prompt_queue.get()
                global_prompt = f"{CONFIG['prompt_base']}, {new_prompt}"
                print(f"Updated prompt: {global_prompt}")

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()