import torch
import ptp_utils
from run_ptp_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl,ldm_stable
from self_guidance import SelfGuidanceCompute,SelfGuidanceEdit,make_self_guidance_controller
import time
x=torch.rand(2,2,requires_grad=True)

y=x+2

z=torch.sum(y)

get_grad=torch.autograd.grad(z,y) 
print(get_grad)
branch=1
if branch==1:
    target="dog"
    # image_path = "./example_images/cat_melon.png"
    # prompt = "A cat is sitting in a pile of watermelons in the trunk"
    # (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    # #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["A cat is running in the forest"]*2
    g_cpu = torch.Generator().manual_seed(123)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    controller = make_self_guidance_controller(prompts, "cat",ldm_stable,mode="right",value=0.3)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,pooled_uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=None)
    # controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    # images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,pooled_uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
    #                             inversion_guidance=True,x_stars=None)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([ images[0],images[1]],text="self_guidance",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图
