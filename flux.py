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
            
            # Get parameters from query string (in the main function, not callback)
            prompt = request.args.get('prompt', 'A cat holding a sign that says hello world')
            negative_prompt = request.args.get('negative_prompt', '')
            guidance_scale = float(request.args.get('guidance_scale', 0.0))
            steps = int(request.args.get('steps', 4))
            seed = int(request.args.get('seed', 42))

            # Print parameters for debugging
            print(request.args)
            
            # Create a list to store the last progress to avoid duplicate messages
            last_progress = [-1]
            
            # Generator function for progress updates
            def progress_generator():
                nonlocal last_progress
                while last_progress[0] < 100:
                    progress = last_progress[0]
                    if progress >= 0:
                        yield f"data: {json.dumps({'progress': progress, 'step': progress * steps // 100, 'total_steps': steps})}\n\n"
                    time.sleep(0.1)  # Prevent busy waiting
            
            # Start progress generator in a separate thread
            import threading
            progress_queue = queue.Queue()
            def run_progress():
                for progress_update in progress_generator():
                    progress_queue.put(progress_update)
                progress_queue.put(None)  # Sentinel value
            
            progress_thread = threading.Thread(target=run_progress)
            progress_thread.start()
            
            # Generate image with callback
            def callback(step, timestep, latents):
                progress = int((step / steps) * 100)
                last_progress[0] = progress
            
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                max_sequence_length=256,
                generator=generator,
                callback=callback
            ).images[0]

            # Wait for progress thread to finish
            progress_thread.join()
            
            # Convert to bytes
            img_io = io.BytesIO()
            image.save(img_io, 'PNG', quality=95)
            img_io.seek(0)
            img_bytes = img_io.getvalue()
            
            # Send completion message
            gen_time = time.time() - start_time
            yield f"data: {json.dumps({'progress': 100, 'status': 'complete', 'time_taken': gen_time})}\n\n"
            
            # In a real implementation, you might want to:
            # 1. Return an image ID here
            # 2. Have a separate endpoint to fetch the image
            # 3. Or implement a proper binary streaming solution
            
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