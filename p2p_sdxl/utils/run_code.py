import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import  DDIMScheduler,DPMSolverMultistepScheduler
from utils.sdxl import sdxl
from utils.run_ptp_utils import Inversion,make_controller,run_and_display
def init_model(model_path="/stable-diffusion-xl-base-1.0",model_dtype="fp16",num_ddim_steps=50,scheduler_type="DDIM"):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if model_dtype=="fp16":
        pipe = sdxl.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif model_dtype=="fp32":
        pipe = sdxl.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=True, variant="fp32")
    elif model_dtype=="fp16":
        pipe = sdxl.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_safetensors=True, variant="bf16")
    if scheduler_type=="DDIM":
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type=="DPMSolver":
        scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler=scheduler
    pipe.to(device)
    #pipe.scheduler=scheduler #用DDIM scheduler
    ldm_stable=pipe
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    inversion = Inversion(ldm_stable,num_ddim_steps,scheduler_type=scheduler_type)
    return ldm_stable,inversion

def run_ptp(prompts,image_path=None,inv_mode:Optional[str]=None,use_replace=False,cross_replace_steps :Optional[float] =0.3,
            self_replace_steps:Optional[float] =0.2,seed:Optional[int]=None,blend_word=None,eq_params=None,guidance_scale=7.5,
            num_ddim_steps=50,model_path="/stable-diffusion-xl-base-1.0",model_dtype="fp16",save_img=True,save_per_img=True,
            masa_control=False,model=None,inversion=None,keep_embdding=False,prox_masa_embdding=False,
            scheduler_type="DDIM",**kwargs):
    ldm_stable=model
    if ldm_stable is None or inversion is None :
        ldm_stable,inversion=init_model(model_path,model_dtype,num_ddim_steps,scheduler_type)
    assert isinstance(prompts,str) or isinstance(prompts,List)
    if isinstance(prompts,str):
        prompts=[prompts]
    cross_replace_steps = {'default_': cross_replace_steps,}
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed()
    if image_path is not None:
        assert isinstance(inv_mode,str)
        (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = inversion.invert(image_path, prompts[0], offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
        assert inv_mode =="proxNPI" or inv_mode =="NPI"
        if  inv_mode =="proxNPI":
            print("Use Proximal Inversion")
            prox_guidance=True
        elif inv_mode =="NPI":
            print("Use Negative-prompt Inversion")
            prox_guidance=False
        else:
            prox_guidance=False
        if keep_embdding==True:
            negative_prompt=None
        if keep_embdding==False:
            prompt_embeds=None
            pooled_prompt_embeds=None
            batch_size=len(prompts)
            if prox_masa_embdding ==True:
                negative_prompt=[prompts[0]]+[""]*(batch_size-1)
            else:
                negative_prompt=[prompts[0]]*batch_size

    else:
        x_t=None
        x_stars=None
        prompt_embeds=None
        pooled_prompt_embeds=None
        prox_guidance=False
        negative_prompt=None
    controller = make_controller(
        prompts,
        ldm_stable,
        num_ddim_steps,
        use_replace,
        cross_replace_steps,
        self_replace_steps,
        blend_word,
        eq_params,
        masa_control,
        **kwargs
    )
    images= run_and_display(
        prompts, 
        controller,
        ldm_stable,
        num_ddim_steps, 
        run_baseline=False, 
        negative_prompt=negative_prompt,
        latent=x_t, 
        uncond_embeddings=prompt_embeds,
        pooled_uncond_embeddings=pooled_prompt_embeds,
        use_old=False,
        one_img=False,
        generator=generator,
        null_inversion=False,
        prox_guidance=prox_guidance,
        x_stars=x_stars,
        guidance_scale=guidance_scale,
        verbose=save_img,
        save_per_img=save_per_img,
        masa_control=masa_control,
        **kwargs
    )
    return images