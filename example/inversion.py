from p2p_sdxl.utils.inversion import Inversion
import torch
from diffusers import DDIMScheduler
from p2p_sdxl.utils.sdxl import sdxl
from p2p_sdxl.utils.ptp_utils import view_images
# 准备模型
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = sdxl.from_pretrained("/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                             use_safetensors=True, variant="fp16", scheduler=scheduler)
model.to(device)
num_ddim_steps = 50
inversion = Inversion(model, num_ddim_steps)
prompt="Photo of a cat riding a bike"
# 进行inversion
(
    (image_gt, image_enc),  # 分别为原图和inversion效果的上界(用vae encode后立刻decode出来)
    x_T,  # 用于去噪的初始噪声
    x_stars,  # inversion过程中各个step的latent组成的List
    prompt_embeds,  # 原prompt的embdding
    pooled_prompt_embeds # 原prompt的pooled embdding (SDXL有两个embdding)
) = inversion.invert(
    image_path="example_images/cat_bike.jpg", #图片路径
    prompt=prompt # 原图prompt
)
# 进行infer
image=sdxl(
    prompt=prompt,
    latents=x_T,
    num_inference_steps=num_ddim_steps,
    guidance_scale=7.5,
    negative_prompt_embeds=prompt_embeds,
    negative_pooled_prompt_embeds=pooled_prompt_embeds,
    same_init=True,
    return_dict=False
)
view_images([image])