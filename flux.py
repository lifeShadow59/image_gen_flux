#!/usr/bin/env python3
import io
import time
from flask import Flask, request, send_file, jsonify
import torch
from diffusers import FluxPipeline
from PIL import Image
import threading

app = Flask(__name__)

# Global pipeline instance (loaded once at startup)
pipe = None
pipe_lock = threading.Lock()
initialized = False

# def initialize_pipeline():
#     global pipe
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA not available. Check GPU drivers")
    
#     pipe = FluxPipeline.from_pretrained(
#         "black-forest-labs/FLUX.1-schnell",
#         torch_dtype=torch.bfloat16,
#         use_safetensors=True
#     ).to("cuda")
    
    # Optimizations
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_model_cpu_offload()




def get_pipeline():
    global pipe
    if pipe is None:
        with pipe_lock:
            if pipe is None:  # Double-check locking pattern
                print("Initializing pipeline...")
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available")
                
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                ).to("cuda")
                print("Pipeline initialized")
    return pipe

with app.app_context():
    get_pipeline()

@app.teardown_appcontext
def shutdown(exception=None):
    global pipe, initialized
    if initialized:
        del pipe
        pipe = None
        initialized = False
        torch.cuda.empty_cache()

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

        # Print parameters for debugging
        print(request.args)
        
        # Generate image
        generator = torch.Generator("cuda").manual_seed(seed)
        print("1")
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            max_sequence_length=256,
            generator=generator
    #       height=768,
    #       width=1360,
    #       num_images_per_prompt=1,
        ).images[0]

        print("2")


        # Print image size for debugging
        print(f"Generated image size: {image.size}")
        
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
    # print("Initializing pipeline...")
    # initialize_pipeline()
    # print("Pipeline ready, starting server...")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, threaded=True)
