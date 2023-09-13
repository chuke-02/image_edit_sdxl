from utils.run_code import run_ptp

prompts=[
"A painting of a city skyline at sunset with the quote 'Dream Big, Achieve More' in elegant calligraphy.",
"A painting of a city skyline at sunset with the quote 'Dream Small, Achieve Less' in elegant calligraphy."
]
# 1.ptp
run_ptp(
    prompts=prompts,
    self_replace_steps = .2,
    cross_replace_steps=.3,
    eq_params = {"words": ("quote",), "values": (1,)},
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0"
)

# 2.ptp+localblend
run_ptp(
    prompts=prompts,
    self_replace_steps = .2,
    cross_replace_steps=.3,
    blend_word = ((('quote',), ("quote",))) ,
    eq_params = {"words": ("quote",), "values": (1,)},
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.4,
    start_blend=0.2,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0"
)