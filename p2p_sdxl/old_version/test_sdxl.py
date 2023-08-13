from diffusers import StableDiffusionXLPipeline,DiffusionPipeline
import torch
import time
from datetime import datetime
base = DiffusionPipeline.from_pretrained(
    "/home/wck/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
high_noise_frac = 0.8
refiner = DiffusionPipeline.from_pretrained(
    "/home/wck/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
for i in range(3):
    prompt = "Photo of a cute fat golden british shorthair riding on a little motorcycle, side-view, full body"
    image = base(
        prompt=prompt,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    t=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"/home/wck/text-to-image-edit/p2p_sdxl/img/sxdl/motorcycle_wrefiner{t}.jpg")
    
for i in range(3):
    prompt = "Photo of a cute fat golden british shorthair riding on a little bicycle, side-view, full body"
    image = base(
        prompt=prompt,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    t=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"/home/wck/text-to-image-edit/p2p_sdxl/img/sxdl/bicycle_wrefiner{t}.jpg")
    

