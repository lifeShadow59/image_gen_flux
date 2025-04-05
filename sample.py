#!/usr/bin/env python3
import sys
import torch
from diffusers import FluxPipeline

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Check GPU drivers", file=sys.stderr)
        sys.exit(1)
    
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
         #   variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        prompt = sys.argv[1] if len(sys.argv) > 1 else "A cat holding a sign that says hello world"
        
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
        
        output_file = f"output_{hash(prompt)}.png"
        image.save(output_file)
        print(f"Success! Image saved to {output_file}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()