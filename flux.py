#!/usr/bin/env python3
import io
import time
import queue
import json
import threading
from flask import Flask, request, send_file, jsonify, Response
import torch
from diffusers import FluxPipeline
from PIL import Image

app = Flask(__name__)

# Global pipeline instance (loaded once at startup)
pipe = None
pipe_lock = threading.Lock()

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

# @app.teardown_appcontext
# def shutdown(exception=None):
#     global pipe
#     if pipe is not None:
#         del pipe
#         pipe = None
#         torch.cuda.empty_cache()

@app.route('/generate_stream', methods=['GET'])
def generate_image_stream():
    # Get parameters within request context
    params = {
        'prompt': request.args.get('prompt', 'A cat holding a sign'),
        'negative_prompt': request.args.get('negative_prompt', ''),
        'guidance_scale': float(request.args.get('guidance_scale', 0.0)),
        'steps': int(request.args.get('steps', 4)),
        'seed': int(request.args.get('seed', 42))
    }

    def generate():
        try:
            start_time = time.time()
            progress_queue = queue.Queue()
            steps = params['steps']
            
            def image_generation_thread():
                try:
                    # Create a generator with the specified seed
                    generator = torch.Generator("cuda").manual_seed(params['seed'])
                    
                    # Generate the image
                    image = pipe(
                        prompt=params['prompt'],
                        negative_prompt=params['negative_prompt'],
                        guidance_scale=params['guidance_scale'],
                        num_inference_steps=steps,
                        max_sequence_length=256,
                        generator=generator
                    ).images[0]
                    
                    # Convert to bytes
                    img_io = io.BytesIO()
                    image.save(img_io, 'PNG')
                    img_io.seek(0)
                    
                    progress_queue.put({
                        'status': 'complete',
                        'image_data': img_io.getvalue().hex(),
                        'time_taken': time.time() - start_time
                    })
                except Exception as e:
                    progress_queue.put({'error': str(e), 'status': 'failed'})

            # Start generation thread
            threading.Thread(target=image_generation_thread).start()

            # Simulate progress updates since FluxPipeline doesn't support callbacks
            for i in range(steps + 1):
                time.sleep(0.5)  # Simulate processing time per step
                progress = int((i / steps) * 100)
                progress_queue.put({
                    'progress': progress,
                    'step': i,
                    'total_steps': steps
                })
                
                if progress >= 100:
                    break

            # Stream updates
            while True:
                update = progress_queue.get()
                if update.get('status') in ['complete', 'failed']:
                    yield f"data: {json.dumps(update)}\n\n"
                    break
                yield f"data: {json.dumps(update)}\n\n"

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