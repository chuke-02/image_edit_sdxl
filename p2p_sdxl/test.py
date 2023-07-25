
import torch
import ptp_utils
from test_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl,ldm_stable

branch =4

if branch==1:#弃用
    prompts = ["A painting of a squirrel eating a burger",
            "A painting of a lion eating a burger"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .8,} #前80%的step替换cross attention
    self_replace_steps = .4 #前40%的step替换self attention
    blend_word = ((('squirrel',), ("lion",))) #将squirrel替换为lion，如果不用LocalBlend就写None
    eq_params = {"words": ("lion",), "values": (1,)} #给某些词几倍权重，如果不用LocalBlend就写None
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,text="localblend")
    #run_baseline指是否跑没有LocalBlend作为对比,在notebook里用的话use_old为True，one_img为False时保存的图片不带timestamp，会覆盖之前的图片，text是保存图片的名字前缀
elif branch==2:#弃用
    image_path = "./example_images/cat.jpg"
    prompt = "Photo of a cat riding on a bicycle"
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

    prompts = ["Photo of a cat riding on a bicycle",
            "Photo of a cat riding on a motorcycle"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .4
    blend_word = ((('bicycle',), ("motorcycle",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": ("motorcycle",), "values": (1,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,use_old=False,one_img=False,generator=g_cpu,inversion=True)
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
elif branch==3:
    target="car"
    image_path = "./example_images/dog.jpg"
    prompt = "Photo of a dog riding on a little bicycle"
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["Photo of a dog riding on a little bicycle", #prompt和prompts[0]要完全一致，否则重建会出问题。句子最后不加句号。
            f"Photo of a dog riding on a little {target}"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null",Notimestamp=True)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图
elif branch==4:
    target="ship"
    prompts = ["Photo of a white cat riding on a little bicycle",
            f"Photo of a white cat riding on a little {target}"]

    g_cpu = torch.Generator().manual_seed(888)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (3,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,text="localblend")
elif branch==5:
    target="plane"
    image_path = "./example_images/dog.jpg"
    prompt = "Photo of a dog riding on a little bicycle"
    ans=[]
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    for i in range(8):
        for j in range(8):
            i0=i*0.1+0.1
            j0=j*0.1+0.1
            
            prompts = ["Photo of a dog riding on a little bicycle",
                    f"Photo of a dog riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': i0,}
            self_replace_steps = j0
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
            images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,inversion=False,verbose=False)
            ans.append(images[1])
    ptp_utils.view_images(ans,num_rows=6,text="cmp")
elif branch==6:
    target="car"
    image_path = "./example_images/dog.jpg"
    prompt = "A"
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)

    prompts = [prompt]
    g_cpu = torch.Generator().manual_seed(12345)
    images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,inversion=False)
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null",Notimestamp=True)
        #ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
#images, _ = run_and_display("motorcycle",controller=EmptyControl(), run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False)
print(1)