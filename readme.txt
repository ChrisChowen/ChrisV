
=============================
Artistic Image Generation with ControlNet
=============================

This project utilizes Stable Diffusion with ControlNet to create artistic images based on ControlNet conditioning and prompts. The generated images can be viewed live through an HTML interface.

Requirements:
--------------
- Python 3.8+
- GPU (Optional but recommended for faster processing)

Installation:
--------------
To install the required dependencies, run the following command:

```bash
pip install torch watchdog diffusers controlnet-aux
```

- `torch`: Required for running the model.
- `watchdog`: Watches for new images in the source directory.
- `diffusers`: Provides the Stable Diffusion and ControlNet models.
- `controlnet-aux`: Provides auxiliary models used for conditioning.

You can also create a virtual environment to isolate the dependencies:

```bash
python -m venv env
source env/bin/activate  # On Linux/macOS
env\Scripts\activate     # On Windows
```

Running the Process:
---------------------
1. **Start the Process**:
   The app script watches a specific directory for new images and processes them. To start the process, run:

   ```bash
   python app.py
   ```

   This will:
   - Watch the `./src` directory for new images.
   - Automatically generate new output images in `./dst` when an image is added.
   - Serve the generated images via an HTTP server at `http://localhost:8000`.

2. **Adding Images**:
   - Place any image file (JPEG or PNG) in the `./src` directory.
   - The script will detect the new image, process it using the ControlNet model, and save the result to the `./dst` directory.

3. **Viewing Results**:
   - Open your browser and go to `http://localhost:8000` to view the original and generated images.
   - The images will be updated in real-time as new steps are processed.

Updating the Prompt Dynamically:
--------------------------------
The prompt can be updated at any time while the script is running. Here's how:
1. **Enter New Prompt**:
   - In the terminal where the script is running, you will be prompted to enter a new text prompt.
   - Type in your new prompt and press `Enter`.

   Example:
   ```
   Enter a new prompt: Surreal landscape, vibrant colors, dreamlike, ethereal lighting
   ```

2. The model will use the new prompt for future image generations. The base prompt is always combined with your dynamic prompt to guide the style.

Directory Structure:
--------------------
- `./src`: Directory where you add images to be processed.
- `./dst`: Directory where the processed images are saved.
- `./html`: Contains the HTML and image files served via the HTTP server.
- `./html/imgs`: Contains the intermediate and final images viewable on the web interface.

Configuration:
--------------
You can configure various settings by editing the `CONFIG` dictionary in the script:

- `src_dir`: Source directory for input images.
- `dst_dir`: Destination directory for output images.
- `html_dir`: Directory for serving images via the web interface.
- `checkpoint`: ControlNet model checkpoint.
- `prompt_base`: Base part of the prompt (fixed).
- `dynamic_prompt`: Dynamic part of the prompt that can be updated live.
- `num_inference_steps`: Number of diffusion steps.
- `controlnet_strength`: Strength of the ControlNet conditioning (0-1).
- `guidance_scale`: Degree to which the image generation should adhere to the prompt.
- `image_resolution`: Resolution of the generated images.
- `seed`: Seed for reproducibility.
- `http_port`: Port number for the HTTP server.

Example:
--------
1. Place an image `input_image.png` in the `./src` folder.
2. Open `http://localhost:8000` in a browser to see the original and generated image.
3. Update the prompt in the terminal to change the style and content of the generated image.

Stopping the Process:
---------------------
To stop the script, press `Ctrl+C` in the terminal.

Enjoy your artistic journey with AI!

=============================
End of File
=============================
