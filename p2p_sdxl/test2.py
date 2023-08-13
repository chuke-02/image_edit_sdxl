from utils.run_ptp_utils import run_ptp

# 3.真实图片编辑(Proximal Inversion)+ptp+localblend
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    image_path="./example_images/cat.png",
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
    use_replace=False
)

# 4.真实图片编辑(Negative Prompt Inversion)+ptp+localblend
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    image_path="./example_images/cat.png",
    inv_mode="NPI", #NPI为Negative Prompt Inversion，proxNPI为Proximal Inversion
    self_replace_steps = .2, #开始替换自注意力的step
    cross_replace_steps=.3, #开始替换交叉的step
    blend_word = ((('bicycle',), ("motorcycle",))) , #保持bicycle和motorcycle并集以外的部分不被编辑
    eq_params = {"words": ("motorcycle",), "values": (1,)}, #将motorcycle对应的cross atten map*1
    seed=12345,
    num_ddim_steps=50, # 采样次数
    guidance_scale=7.5, # 去噪时的CFG(加噪时的CFG为1)
    mask_threshold=0.4,  # localblend确定mask时使用的阈值
    start_blend=0.2, # 从20%的step开始使用localblend
    use_replace=False # 使用refine,而非replace
)