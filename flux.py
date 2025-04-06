#!/usr/bin/env python3
import io
import time
from flask import Flask, request, send_file, jsonify, Response
import torch
from diffusers import FluxPipeline
from PIL import Image
import threading
import json

app = Flask(__name__)

# Global pipeline instance (loaded once at startup)
pipe = None
pipe_lock = threading.Lock()
initialized = False

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

def generate_with_progress(prompt, negative_prompt, guidance_scale, steps, seed):
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Create a callback function to report progress
    def callback(step, timestep, latents):
        progress = int((step / steps) * 100)
        yield f"data: {json.dumps({'progress': progress, 'step': step, 'total_steps': steps})}\n\n"
    
    # Generate image with callback
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=256,
        generator=generator,
        callback=lambda step, timestep, latents: callback(step, timestep, latents)
    ).images[0]
    
    return image

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

@app.route('/generate_stream', methods=['GET'])
def generate_image_stream():
    def generate():
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
            
            # Generate image with progress reporting
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Create a list to store the last progress to avoid duplicate messages
            last_progress = [-1]
            
            def callback(step, timestep, latents):
                progress = int((step / steps) * 100)
                if progress != last_progress[0]:
                    last_progress[0] = progress
                    yield f"data: {json.dumps({'progress': progress, 'step': step, 'total_steps': steps})}\n\n"
            
            # Start image generation
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=256,
                generator=generator,
                callback=lambda step, timestep, latents: [cb for cb in callback(step, timestep, latents)]
            ).images[0]

            # Convert to bytes
            img_io = io.BytesIO()
            image.save(img_io, 'PNG', quality=95)
            img_io.seek(0)
            img_bytes = img_io.getvalue()
            
            # Send completion message with image data
            gen_time = time.time() - start_time
            yield f"data: {json.dumps({'progress': 100, 'status': 'complete', 'image_size': len(img_bytes), 'time_taken': gen_time})}\n\n"
            
            # Send the image data (encoded as base64 or in chunks)
            # Here we'll just send a message that generation is complete
            # In a real implementation, you might want to send the image data differently
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'status': 'failed'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ready",
        "cuda_available": torch.cuda.is_available(),
        "device": str(pipe.device) if pipe else "uninitialized"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)