from utils.run_code import run_ptp, init_model
from utils.ptp_utils import view_images
prompts=["a dog is jumping on the road", "a dog is sitting on the road"]
prompt=prompts[0]+"\n"+prompts[1]
grid_dict={"x_title":"","y_title":"","font_size":100}
grid_dict["x_text_list"]=['reconstruction\nDDIM','edit\nDDIM','reconstruction\nDPMsolver++','edit\nDPMsolver++']
grid_dict["y_text_list"]=[
                          "NPI\n",
                          
                          "proxNPI\n"
                          ]
grid_dict['title']=prompt+""
grid_dict['shift_y']=-330,0
ans=[]
scheduler_type="DDIM"#DPMSolver
scheduler_type2="DPMSolver"

img =run_ptp(
    prompts=prompts,
    image_path="./example_images/dog_jumping.jpg",
    inv_mode="NPI",
    self_replace_steps=0.0,  # 不使用ptp
    cross_replace_steps=0.0,  # 不使用ptp
    blend_word=((("dog",), ("dog",))),
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
    prox_masa_embdding=False,
    save_img=False,
    scheduler_type=scheduler_type
)
ans+=[img[0]]+[img[1]]
img =run_ptp(
    prompts=prompts,
    image_path="./example_images/dog_jumping.jpg",
    inv_mode="NPI",
    self_replace_steps=0.0,  # 不使用ptp
    cross_replace_steps=0.0,  # 不使用ptp
    blend_word=((("dog",), ("dog",))),
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
    prox_masa_embdding=False,
    save_img=False,
    scheduler_type=scheduler_type2
)
ans+=[img[0]]+[img[1]]

img =run_ptp(
    prompts=prompts,
    image_path="./example_images/dog_jumping.jpg",
    inv_mode="proxNPI",
    self_replace_steps=0.0,  # 不使用ptp
    cross_replace_steps=0.0,  # 不使用ptp
    blend_word=((("dog",), ("dog",))),
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
    prox_masa_embdding=False,
    save_img=False,
    scheduler_type=scheduler_type
)
ans+=[img[0]]+[img[1]]
img =run_ptp(
    prompts=prompts,
    image_path="./example_images/dog_jumping.jpg",
    inv_mode="proxNPI",
    self_replace_steps=0.0,  # 不使用ptp
    cross_replace_steps=0.0,  # 不使用ptp
    blend_word=((("dog",), ("dog",))),
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
    prox_masa_embdding=False,
    save_img=False,
    scheduler_type=scheduler_type2
)
ans+=[img[0]]+[img[1]]
view_images(ans,num_rows=2,text="masa_uncond_cmp",grid_dict=grid_dict
)