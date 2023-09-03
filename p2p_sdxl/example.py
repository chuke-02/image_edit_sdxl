from utils.run_code import run_ptp


# 1.ptp
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    self_replace_steps = .2,
    cross_replace_steps=.3,
    eq_params = {"words": ("motorcycle",), "values": (1,)},
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    use_replace=False,
    model_path="stabilityai/stable-diffusion-xl-base-1.0"
)

# 2.ptp+localblend
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    self_replace_steps = .2,
    cross_replace_steps=.3,
    blend_word = ((('bicycle',), ("motorcycle",))) ,
    eq_params = {"words": ("motorcycle",), "values": (1,)},
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.4,
    start_blend=0.2,
    use_replace=False,
    model_path="stabilityai/stable-diffusion-xl-base-1.0"
)

# 3.真实图片编辑(Proximal Inversion)+ptp+localblend
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    image_path="./example_images/cat_bike3.jpg",
    inv_mode="proxNPI",
    self_replace_steps = .2,
    cross_replace_steps=.3,
    blend_word = ((('bicycle',), ("motorcycle",))) ,
    eq_params = {"words": ("motorcycle",), "values": (1,)},
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.4,
    start_blend=0.2,
    use_replace=False,
    model_path="stabilityai/stable-diffusion-xl-base-1.0"
)

# 4.真实图片编辑(Negative Prompt Inversion)+ptp+localblend
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    image_path="./example_images/cat_bike3.jpg",
    inv_mode="NPI", #NPI为Negative Prompt Inversion，proxNPI为Proximal Inversion
    self_replace_steps = .2, # 前20%step替换自注意力
    cross_replace_steps=.3, # 前30%step替换交叉注意力
    blend_word = ((('bicycle',), ("motorcycle",))) , #保持bicycle和motorcycle并集以外的部分不被编辑
    eq_params = {"words": ("motorcycle",), "values": (1,)}, #将motorcycle对应的cross atten map*1
    seed=12345,
    num_ddim_steps=50, # 采样次数
    guidance_scale=7.5, # 去噪时的CFG(加噪时的CFG为1)
    mask_threshold=0.4,  # localblend确定mask时使用的阈值
    start_blend=0.2, # 从20%的step开始使用localblend
    use_replace=False, # 使用refine,而非replace
    model_path="stabilityai/stable-diffusion-xl-base-1.0"
)

# 5.masa control
run_ptp(
    prompts=["a cat is running",
        "two cats are running"],
    self_replace_steps =0.0, #不使用ptp
    cross_replace_steps=0.0, #不使用ptp
    blend_word = ((('cat',), ("cat",))) ,
    eq_params = None,
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.5,
    start_blend=0.2,
    use_replace=False,
    model_path="stabilityai/stable-diffusion-xl-base-1.0",
    masa_control=True,  # 开启masa control
    masa_mask=False, # 是否使用基于mask的masa control，如果使用，要设定对应的blend_word
    masa_start_step=10, #从第step 10开始进行masa control（替换self attention的 kv） 
    masa_start_layer=45, #从unet的第45
    x_t_replace=False, #True的话启用localblend，False的话不用localblend
    )

run_ptp(
    prompts=["a cat is running",
        "a cat is laying"],
    self_replace_steps =0.0, #不使用ptp
    cross_replace_steps=0.0, #不使用ptp
    blend_word = ((('cat',), ("cat",))) ,
    eq_params = None,
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.5,
    start_blend=0.2,
    use_replace=False,
    model_path="stabilityai/stable-diffusion-xl-base-1.0",
    masa_control=True,  # 开启masa control
    masa_mask=False, # 是否使用基于mask的masa control，如果使用，要设定对应的blend_word
    masa_start_step=10, #从第step 10开始进行masa control（替换self attention的 kv） 
    masa_start_layer=45, #从unet的第45
    x_t_replace=False, #True的话启用localblend，False的话不用localblend
    )

