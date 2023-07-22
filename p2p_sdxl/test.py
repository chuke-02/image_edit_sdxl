
import torch
import ptp_utils
from test_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion,EmptyControl

branch =4
if branch==1:
    prompts = ["A painting of a squirrel eating a burger",
            "A painting of a lion eating a burger"]
    # controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
    # _ = run_and_display(prompts, controller, latent=None, run_baseline=False,use_old=False,one_img=True)
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .4
    blend_word = ((('squirrel',), ("lion",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": ("lion",), "values": (1,)} # amplify attention to the word "tiger" by *2 
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu,text="localblend")
elif branch==2:
    image_path = "./example_images/cat.jpg"
    prompt = "Photo of a cat riding on a bicycle"
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

    prompts = ["Photo of a cat riding on a bicycle",
            "Photo of a cat riding on a motorcycle"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .4
    blend_word = ((('bicycle',), ("motorcycle",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": ("motorcycle",), "values": (1,)} # amplify attention to the word "tiger" by *2 
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,use_old=False,one_img=False,generator=g_cpu,inversion=True)
    ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
elif branch==3:
    target="car"
    image_path = "./example_images/dog.jpg"
    prompt = "Photo of a dog riding on a little bicycle"
    (image_gt, image_enc), x_t, prompt_embeds,pooled_prompt_embeds = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True,train_free=True)

    prompts = ["Photo of a dog riding on a little bicycle",
            f"Photo of a dog riding on a little {target}"]
    g_cpu = torch.Generator().manual_seed(12345)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (2,)} # amplify attention to the word "tiger" by *2 
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=prompt_embeds,pooled_uncond_embeddings=pooled_prompt_embeds,use_old=False,one_img=True,generator=g_cpu,inversion=False)
    #ptp_utils.view_images([image_gt, image_enc, images[0]],text="null")
elif branch==4:
    target="ship"
    prompts = ["Photo of a cat riding on a little bicycle",
            f"Photo of a cat riding on a little {target}"]
    # controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
    # _ = run_and_display(prompts, controller, latent=None, run_baseline=False,use_old=False,one_img=True)
    g_cpu = torch.Generator().manual_seed(12)
    cross_replace_steps = {'default_': .3,}
    self_replace_steps = .2
    blend_word = ((('bicycle',), (f"{target}",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    eq_params = {"words": (f"{target}",), "values": (3,)} # amplify attention to the word "tiger" by *2 
    controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps,blend_word,eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=True,generator=g_cpu,text="localblend")
#images, _ = run_and_display("motorcycle",controller=EmptyControl(), run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False)
print(1)