
import torch
from test_utils import AttentionReplace,run_and_display,NUM_DDIM_STEPS,make_controller,null_inversion
branch = 2
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
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=None, uncond_embeddings=None,use_old=False,one_img=False,generator=g_cpu)
elif branch==2:
    image_path = "./example_images/cat.jpg"
    prompt = "Photo of a cat riding on a bicycle."
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

    prompts = ["Photo of a cat riding on a bicycle.",
            "Photo of a cat riding on a motorcycle."]

print(1)