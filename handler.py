import runpod
import torch
from diffusers import DiffusionPipeline
import base64
from io import BytesIO

pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        print("Loading Qwen-Image-2512...")
        pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-2512",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("Model loaded!")
    return pipe

def handler(job):
    try:
        input_data = job["input"]
        prompt = input_data.get("prompt", "")
        
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Enhanced prompt
        prompt = prompt + ", Ultra HD, 4K, cinematic composition."
        
        # Generate
        pipeline = load_pipeline()
        image = pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=4.0
        ).images[0]
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
