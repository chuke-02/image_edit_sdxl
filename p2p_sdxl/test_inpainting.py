import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from PIL import Image,ImageOps
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "/home/wck/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = load_image(img_url).convert("RGB")
# mask_image = load_image(mask_url).convert("RGB")
init_image = Image.open('/home/wck/text-to-image-edit/p2p_sdxl/example_images/dog.jpg')
mask_image= Image.open('/home/wck/text-to-image-edit/p2p_sdxl/example_images/dog_dog_mask.jpg')
mask_image=ImageOps.invert(mask_image)
init_image.save('init_image.jpg', 'JPEG')
mask_image.save('mask_image.jpg', 'JPEG')
prompt = "Photo of a dog riding on a motocycle"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]

# 保存为JPEG格式 
image.save('sdxl-inpainting-replace-except_dog-to-motocycle.jpg', 'JPEG')
print(1)