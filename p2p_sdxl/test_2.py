import torch
import ptp_utils
from run_ptp_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl,ldm_stable
#import importlib

if True:
    ww=1
    for w in range(1,2):
        #test_utils = importlib.reload(test_utils)
        
        target="motorcycle"
        image_path = f"./example_images/cat_bike{ww}.jpg"
        prompt = "Photo of a cat riding on a little bicycle"
        ans=[]
        grid_dict={"x_title":"self","y_title":"      cross      ","font_size":80,"num_decimals":2,"shift_y":(-250,0)}
        x_text_list=["original",'no_p2p']
        y_text_list=[]
        for i in range(3):
            x_text_list.append(i*0.05+0.1)
        y_text_list.append("0.30\n(left two is 0.00)")
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
            if j==0:
                ans.append(images[0])
            ans.append(images[1])
            print(ww)
        ptp_utils.view_images(ans,num_rows=1,text="cmp",grid_dict=grid_dict)