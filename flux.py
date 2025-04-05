run_flux1.py
#!/usr/bin/env python3
import io
import time
from flask import Flask, request, send_file, jsonify
import torch
from diffusers import FluxPipeline
from PIL import Image

app = Flask(__name__)

# Global pipeline instance (loaded once at startup)
pipe = None

def initialize_pipeline():
    global pipe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check GPU drivers")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
    
    # Optimizations
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

@app.route('/generate', methods=['GET'])
def generate_image():
    try:
        start_time = time.time()
        
        # Get parameters from query string
        prompt = request.args.get('prompt', 'A cat holding a sign that says hello world')
        negative_prompt = request.args.get('negative_prompt', '')
        guidance_scale = float(request.args.get('guidance_scale', 0.0))
        steps = int(request.args.get('steps', 4))
        seed = int(request.args.get('seed', 42))
        
        # Generate image
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            max_sequence_length=256,
            generator=generator
        ).images[0]
        
        # Convert to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        
        # Log performance
        gen_time = time.time() - start_time
        print(f"Generated image in {gen_time:.2f}s | Prompt: {prompt}")
        
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ready",
        "cuda_available": torch.cuda.is_available(),
        "device": str(pipe.device) if pipe else "uninitialized"
    })

if __name__ == '__main__':
    # Initialize pipeline before serving
    print("Initializing pipeline...")
    initialize_pipeline()
    print("Pipeline ready, starting server...")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, threaded=True)
