from utils.run_ptp_utils import run_ptp
from utils.ptp_utils import view_images
prompts=["a black dog is running",
            "a dog is sitting"]
text_list=["motorcycle","car","train","plane","ship"]
prompt=prompts[0]+"\n"+prompts[1]
grid_dict={"x_title":"","y_title":"","font_size":100}
grid_dict["x_text_list"]=['running','sitting']
grid_dict["y_text_list"]=['None','masa_controll']
grid_dict['title']=prompt
grid_dict['shift_y']=-300,0
img=run_ptp(
    prompts=prompts,
    #image_path="./example_images/dog_bike3.jpg",
    #inv_mode="NPI",
    self_replace_steps = 0.0,
    cross_replace_steps=0.0,
    blend_word = ((('dog',), ("dog",))) ,
    eq_params = None,
    seed=1,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.5,
    start_blend=0.2,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0",
    masa_control=False,
    x_t_replace=False,
    save_img=False
)
img1=run_ptp(
    prompts=prompts,
    #image_path="./example_images/dog_bike3.jpg",
    #inv_mode="NPI",
    self_replace_steps = 0.0,
    cross_replace_steps=0.0,
    blend_word = ((('dog',), ("dog",))) ,
    eq_params = None,
    seed=1,
    num_ddim_steps=50,
    guidance_scale=7.5,
    mask_threshold=0.5,
    start_blend=0.2,
    use_replace=False,
    model_path="/stable-diffusion-xl-base-1.0",
    masa_control=True,
    x_t_replace=False,
    save_img=False
)
view_images([img[0]]+[img[1]]+[img1[0]]+[img1[1]],num_rows=2,text="masa",grid_dict=grid_dict)
# run_ptp(
#     prompts=["Photo of a dog riding on a little bicycle",
#             "Photo of a cat flying on a little bicycle"],
#     image_path="./example_images/cat_bike3.jpg",
#     inv_mode="NPI",
#     self_replace_steps = 1.0,
#     cross_replace_steps=1.0,
#     blend_word = ((('cat',), ("cat",))) ,
#     eq_params = None,
#     seed=12345,
#     num_ddim_steps=50,
#     guidance_scale=7.5,
#     mask_threshold=0.4,
#     start_blend=0.2,
#     use_replace=False,
#     model_path="/stable-diffusion-xl-base-1.0",
#     masa_control=False,
#     x_t_replace=False
# )