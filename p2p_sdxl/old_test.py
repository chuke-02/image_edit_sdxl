# import torch
# import ptp_utils
# from run_ptp_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl,ldm_stable
# #import importlib

# if True:
#     ww=1
#     for w in range(1,2):
#         #test_utils = importlib.reload(test_utils)
        
#         target="motorcycle"
#         image_path = f"./example_images/cat_bike{ww}.jpg"
#         prompt = "Photo of a cat riding on a little bicycle"
#         ans=[]
#         grid_dict={"x_title":"self","y_title":"      cross      ","font_size":80,"num_decimals":2,"shift_y":(-250,0)}
#         x_text_list=["original",'no_p2p']
#         y_text_list=[]
#         for i in range(3):
#             x_text_list.append(i*0.05+0.1)
#         y_text_list.append("0.30\n(left two is 0.00)")
#         grid_dict["x_text_list"]=x_text_list
#         grid_dict["y_text_list"]=y_text_list
#         (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
#         for j in range(4):
#             i0=0.3
#             j0=j*0.05+0.05
#             prompts = ["Photo of a cat riding on a little bicycle",
#                     f"Photo of a cat riding on a little {target}"]
#             g_cpu = torch.Generator().manual_seed(12345)
#             cross_replace_steps = {'default_': i0,}
#             self_replace_steps = j0
#             blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
#             eq_params = {"words": (f"{target}",), "values": (2,)}
#             controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
#             if j==0:
#                 images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False,verbose=False)
#             else:
            
#                 images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False,verbose=False)
#             if j==0:
#                 ans.append(images[0])
#             ans.append(images[1])
#             print(ww)
#         ptp_utils.view_images(ans,num_rows=1,text="cmp",grid_dict=grid_dict)


import torch
from utils import ptp_utils
from utils.run_ptp_utils import run_ptp
NUM_DDIM_STEPS=50
import time
branch =5

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
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,use_old=False,one_img=False,generator=g_cpu,null_inversion=True)
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
elif branch==3:
    target="green"
    image_path = "./example_images/dogc.jpg"
    prompt = "A cat is sitting in a pile of watermelons in the trunk"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["a photo of dog running in gray forest",
            "a photo of dog running in green forest"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('trunk',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)}
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=x_stars)
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t,
                                     uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,
                                     use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars)
    
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new_mask, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars,use_localblend_mask=True)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt,  images[0],images_new[0]],text="null",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图
    grid_dict={"x_title":"","y_title":"","font_size":100}
    grid_dict["x_text_list"]=['NPI','proxNPI','proxNPI\n+LocalBlendmask']
    grid_dict["y_text_list"]=['reconstruction','edited image']
    grid_dict['title']=prompts
    grid_dict['shift_y']=-300,0
    ptp_utils.view_images([images[0],images_new[0],images_new_mask[0],images[1],images_new[1],images_new_mask[1]],num_rows=2,text="cmp",grid_dict=grid_dict)
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
    grid_dict={"x_title":"self","y_title":"cross","font_size":200}
    x_text_list=[]
    y_text_list=[]
    for i in range(9):
        x_text_list.append(i*0.1)
        y_text_list.append(i*0.1)
    grid_dict["x_text_list"]=x_text_list
    grid_dict["y_text_list"]=y_text_list
    #(image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    for i in range(9):
        for j in range(9):
            i0=i*0.1
            j0=j*0.1
            prompts = ["Photo of a dog riding on a little bicycle",
                    f"Photo of a dog riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': i0,}
            self_replace_steps = j0
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            #controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
            images=run_ptp(
                prompts=prompts,
                image_path=image_path,
                inv_mode="proxNPI",
                self_replace_steps = j0,
                cross_replace_steps=i0,
                blend_word = ((('bicycle',), (f"{target}",))) ,
                eq_params = {"words": (f"{target}",), "values": (2,)},
                seed=12345,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                save_img=False,
                save_per_img=False
                )
            ans.append(images[1])
    ptp_utils.view_images(ans,num_rows=9,text="cmp_woInverse",grid_dict=grid_dict)
elif branch==6:
    target="car"
    image_path = "./example_images/dog.jpg"
    prompt = "A"
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)

    prompts = [prompt]
    g_cpu = torch.Generator().manual_seed(12345)
    images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False)
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null",Notimestamp=True)
elif branch==7:# test refine
    target="ship"
    prompts = ["Photo of a white cat riding on a little bicycle",
            f"Painting of a balck cat riding on a little car on the sea"]

    g_cpu = torch.Generator().manual_seed(888)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    #blend_word = ((('bicycle',), (f"car",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"car",), "values": (3,)}
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,None,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,text="refine")
        #ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
elif branch==8:# cas replace
    image_path = "./example_images/dogc.jpg"
    prompt = "a photo of dog running in gray forest"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)

    prompts = ["a photo of dog running in gray forest",
            "a photo of cat running in gray forest"]

    g_cpu = torch.Generator().manual_seed(888)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('dog',), (f"cat",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"cat",), "values": (2,)}
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t,
                                 uncond_embeddings=prompt_embeds,
                                 pooled_uncond_embeddings=pooled_prompt_embeds,
                                 use_old=False,one_img=False,generator=None,text="cas",folder="cas",
                                 inversion_guidance=True,x_stars=x_stars)
    ptp_utils.view_images([images[0],images[1]],text="cas_oneimg",subfolder="cas")
elif branch==9:
    prompts = ["a photo of dog running in gray forest",
            "a photo of dog running in golden wheat field"]

    g_cpu = torch.Generator().manual_seed(888)
    #blend_word = ((('gray',), (f"golden",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    #eq_params = {"words": (f"golden",), "values": (2,)}
    #controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,None,None)
    images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=None, uncond_embeddings=None,pooled_uncond_embeddings=None,use_old=False,one_img=False,generator=None,text="cas",folder="cas")
    ptp_utils.view_images([images[1]],text="generate")
elif branch==10:#测试新图
    target="motorcycle"
    image_path = "./example_images/cat_bike1.jpg"
    prompt = "Photo of a cat riding on a little bicycle"
    ans=[]
    grid_dict={"x_title":"self","y_title":"cross","font_size":200}
    x_text_list=[]
    y_text_list=[]
    for i in range(9):
        x_text_list.append(i*0.1)
        y_text_list.append(i*0.1)
    grid_dict["x_text_list"]=x_text_list
    grid_dict["y_text_list"]=y_text_list
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    for i in range(9):
        for j in range(9):
            i0=i*0.1
            j0=j*0.1
            prompts = ["Photo of a cat riding on a little bicycle",
                    f"Photo of a cat riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': i0,}
            self_replace_steps = j0
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
            images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        )
            ans.append(images[1])
    ptp_utils.view_images(ans,num_rows=9,text="cmp",grid_dict=grid_dict)
elif branch==11:#连续测试
    for w in range(1,2):
        from p2p_sdxl.run_ptp_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl,ldm_stable
        target="motorcycle"
        image_path = f"./example_images/cat_bike2.jpg"
        prompt = "Photo of a cat riding on a little bicycle"
        ans=[]
        grid_dict={"x_title":"self","y_title":"cross","font_size":100,"num_decimals":2}
        x_text_list=['no_p2p']
        y_text_list=[]
        for i in range(3):
            x_text_list.append(i*0.05+0.1)
        y_text_list.append("0.3(1st:0.0)")
        grid_dict["x_text_list"]=x_text_list
        grid_dict["y_text_list"]=y_text_list
        (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
        for j in range(4):
            i0=0.3
            j0=j*0.05+0.05
            prompts = ["Photo of a cat riding on a little bicycle",
                    f"Photo of a cat riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': i0,}
            self_replace_steps = j0
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
            if j==0:
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False,verbose=False)
            else:
                images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,null_inversion=False,verbose=False)
            ans.append(images[1])
            print(w,j)
        ptp_utils.view_images(ans,num_rows=1,text="cmp",grid_dict=grid_dict)
elif branch==12:#测试新inversion
    target="motorcycle"
    image_path = "./example_images/cat_bike1.jpg"
    prompt = "Photo of a cat riding on a little bicycle"
    ans=[]
    grid_dict={"x_title":"self","y_title":"cross","font_size":200}
    x_text_list=[]
    y_text_list=[]
    for i in range(9):
        x_text_list.append(i*0.1)
        y_text_list.append(i*0.1)
    grid_dict["x_text_list"]=x_text_list
    grid_dict["y_text_list"]=y_text_list
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    for i in range(9):
        for j in range(9):
            i0=i*0.1
            j0=j*0.1
            prompts = ["Photo of a cat riding on a little bicycle",
                    f"Photo of a cat riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': i0,}
            self_replace_steps = j0
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
            images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=True
                                        )
            ans.append(images[1])
    ptp_utils.view_images(ans,num_rows=9,text="cmp",grid_dict=grid_dict)
elif branch==13:#测试时间开销
    target="motorcycle"
    image_path = "./example_images/cat_bike1.jpg"
    prompt = "Photo of a cat riding on a little bicycle"
    ans=[]
    grid_dict={"x_title":"self","y_title":"cross","font_size":200}
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)
    time_info=["negative_prompt_inversion",
               "inversion_guidance",
               "inversion_guidance+reconstruction_guidance",
               "inversion_guidance+reconstruction_guidance+p2p",
               "inversion_guidance+reconstruction_guidance+p2p+Localblend"]
    for kk in range(5):
        start_time = time.time()
        for j in range(10):
            prompts = ["Photo of a cat riding on a little bicycle",
                    f"Photo of a cat riding on a little {target}"]
            g_cpu = torch.Generator().manual_seed(12345)
            cross_replace_steps = {'default_': 0.3,}
            self_replace_steps = 0.2
            blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
            eq_params = {"words": (f"{target}",), "values": (2,)}
            
            if kk==0:
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=False,reconstruction_guidance=False,
                                        text=time_info[kk]
                                        )
            elif kk==1:
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=True,reconstruction_guidance=False,
                                        text=time_info[kk]
                                        )
            elif kk==2:
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=True,reconstruction_guidance=True,
                                        text=time_info[kk]
                                        )
            elif kk==3:
                controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,None,None)
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=True,reconstruction_guidance=True,
                                        text=time_info[kk]
                                        )    
            elif kk==4:
                controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
                images, _ = run_and_display(prompts, EmptyControl(), run_baseline=False, latent=x_t,
                                        uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,
                                        generator=g_cpu,null_inversion=False,verbose=False,
                                        inversion_guidance=True,reconstruction_guidance=True,
                                        text=time_info[kk]
                                        )                
        end_time = time.time()
        print(f"{time_info[kk]}: {end_time - start_time}s")
elif branch==14:
    target="tiger"
    image_path = "./example_images/gnochi_mirror.jpeg"
    prompt = "a cat sitting next to a mirror"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["a cat sitting next to a mirror", #prompt和prompts[0]要完全一致，否则重建会出问题。句子最后不加句号。
            f"a {target} sitting next to a mirror"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('cat',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=x_stars)
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt, images[0],images_new[0]],text="null",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图

    #ptp_utils.view_images(ans,num_rows=9,text="cmp",grid_dict=grid_dict)
elif branch==15:
    #target="tiger"
    image_path = "./example_images/gnochi_mirror.jpeg"
    prompt = "a cat sitting next to a mirror"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["a cat sitting next to a mirror", #prompt和prompts[0]要完全一致，否则重建会出问题。句子最后不加句号。
            f"a silver cat sculpture sitting next to a mirror"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('cat',), (f"cat",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": ("silver", 'sculpture', ), "values": (1,2,)}
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=x_stars)
    controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt, images[0],images_new[0]],text="null",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图

elif branch==16:
    target="dog"
    image_path = "./example_images/cat_melon.png"
    prompt = "A cat is sitting in a pile of watermelons in the trunk"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["A cat is sitting in a pile of watermelons in the trunk", 
                     f"A dog is sitting in a pile of watermelons in the trunk"]
    g_cpu = torch.Generator().manual_seed(123)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('cat',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=x_stars)
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt, images[0],images_new[0]],text="null",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图

    #ptp_utils.view_images(ans,num_rows=9,text="cmp",grid_dict=grid_dict)
elif branch==17:
    target="umbrellas"
    image_path = "./example_images/tree.jpg"
    prompt = "A group of elephants on the African grasslands, with several trees in the background"
    (image_gt, image_enc), x_t,x_stars, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True,all_latents=True)
    #train_free为True时用Negative prompt Inversion（速度快，效果不完美），为False时用Null Text Inversion（要训练uncond_embeddings，所以速度慢、效果好，但是有bug）
    prompts = ["A group of elephants on the African grasslands, with several trees in the background", 
                     f"A group of elephants on the African grasslands, with several umbrellas in the background"]
    g_cpu = torch.Generator().manual_seed(123)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('trees',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)}
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=False,x_stars=x_stars)
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images_new, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=False,generator=g_cpu,null_inversion=False,
                                inversion_guidance=True,x_stars=x_stars)
    #run_and_display默认会保存生成的图片，在img/下
    #如果用Null Text Inversion要设置null_inversion=True，然而这个功能有bug，所以设置null_inversion=False就好
    ptp_utils.view_images([image_gt, images[0],images_new[0]],text="null",Notimestamp=False)#保存一张图，左中右分别为：原图，VAE Decode后的图，重建的图

 
#images, _ = run_and_display("motorcycle",controller=EmptyControl(), run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False)
print(1)