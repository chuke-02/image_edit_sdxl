from utils.run_code import run_ptp, init_model

run_ptp(
    prompts=["a cat is running", "a cat is jumping"],
    image_path="./example_images/cat_run.jpg",
    inv_mode="proxNPI",
    self_replace_steps=0.0,  # 不使用ptp
    cross_replace_steps=0.0,  # 不使用ptp
    blend_word=((("cat",), ("cat",))),
    eq_params=None,
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.5,
    start_blend=0.2,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0",
    masa_control=True,  # 开启masa control
    masa_mask=False,  # 是否使用基于mask的masa control，如果使用，要设定对应的blend_word
    masa_start_step=10,  # 从第step 10开始进行masa control（替换self attention的 kv）
    masa_start_layer=45,  # 从unet的第45
    x_t_replace=False,  # True的话启用localblend，False的话不用localblend
)
