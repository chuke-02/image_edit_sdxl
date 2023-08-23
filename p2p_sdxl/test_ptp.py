from utils.run_code import run_ptp
run_ptp(
    prompts=["A beach scene with surfboards lined up and a sign that reads 'Catch the Perfect Wave'.",
            "A beach scene with surfboards lined up and a sign that reads 'Catch the Best Wave'."],
    #image_path="./example_images/sand.png",
    #inv_mode="proxNPI",
    self_replace_steps = .2,
    cross_replace_steps=.3,
    blend_word = ((('sign',), ("sign",))) ,
    eq_params = {"words": ("sign",), "values": (1,)},
    seed=1234567,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.4,
    start_blend=0.2,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0"
)