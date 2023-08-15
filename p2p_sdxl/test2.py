from utils.run_ptp_utils import run_ptp
from utils.ptp_utils import view_images
branch =2
if branch==1:
    ans=[]
    text_list=["motorcycle","car","train","plane","ship"]
    prompt="Photo of a dog riding on a little bicycle\nself_replace_steps=0.2\nmobile SAM mask"
    grid_dict={"x_title":"","y_title":"","font_size":100}
    grid_dict["x_text_list"]=['NPI','proxNPI']
    grid_dict["y_text_list"]=['reconstruction']
    grid_dict['title']=prompt
    grid_dict['shift_y']=-300,0
    for i,text in enumerate(text_list):
    # 3.真实图片编辑( )+ptp+localblend
        img=run_ptp(
            prompts=["Photo of a dog riding on a little bicycle",
                    f"Photo of a dog riding on a little {text}"],
            image_path="./example_images/dog.jpg",
            inv_mode="NPI",
            self_replace_steps = .2,
            cross_replace_steps=.3,
            blend_word = ((('bicycle',), (f"{text}",))) ,
            eq_params = {"words": (f"{text}",), "values": (2,)},
            seed=12345,
            num_ddim_steps=50,
            guidance_scale=7.5,
            mask_threshold=0.5,
            start_blend=0.2,
            use_replace=False,
            save_img=False
        )

    # 4.真实图片编辑(  Inversion)+ptp+localblend
        img1=run_ptp(
            prompts=["Photo of a dog riding on a little bicycle",
                    f"Photo of a dog riding on a little {text}"],
            image_path="./example_images/dog.jpg",
            inv_mode="proxNPI", #NPI为Negative Prompt Inversion，proxNPI为Proximal Inversion
            self_replace_steps = .2, #开始替换自注意力的step
            cross_replace_steps=.3, #开始替换交叉的step
            blend_word = ((('bicycle',), (f"{text}",))) , #保持bicycle和motorcycle并集以外的部分不被编辑
            eq_params = {"words": (f"{text}",), "values": (2,)},#将motorcycle对应的cross atten map*1
            seed=12345,
            num_ddim_steps=50, # 采样次数
            guidance_scale=7.5, # 去噪时的CFG(加噪时的CFG为1)
            mask_threshold=0.5,  # localblend确定mask时使用的阈值
            start_blend=0.2, # 从20%的step开始使用localblend
            use_replace=False, # 使用refine,而非replace,
            save_img=False
        )
        grid_dict["y_text_list"]+=[f"bicycle->{text}"]
        if i==0:
            ans+=[img[0],img1[0]]
        ans+=[img[1],img1[1]]

    view_images(ans,num_rows=len(grid_dict["y_text_list"]),text="cmp",grid_dict=grid_dict)
elif branch==2:
    ans=[]
    text_list=["cat","lion","squirrel","tiger","rabbit"]
    prompt="Photo of a dog riding on a little bicycle\nself_replace_steps=0.2"#
    grid_dict={"x_title":"","y_title":"","font_size":100}
    grid_dict["x_text_list"]=['NPI','proxNPI']
    grid_dict["y_text_list"]=['reconstruction']
    grid_dict['title']=prompt
    grid_dict['shift_y']=-300,0
    for i,text in enumerate(text_list):
    # 3.真实图片编辑( )+ptp+localblend
        img=run_ptp(
            prompts=["Photo of a dog riding on a little bicycle",
                    f"Photo of a {text} riding on a little bicycle"],
            image_path="./example_images/dog.jpg",
            inv_mode="NPI",
            self_replace_steps = .2,
            cross_replace_steps=.3,
            blend_word = ((('dog',), (f"{text}",))) ,
            eq_params = {"words": (f"{text}",), "values": (2,)},
            seed=12345,
            num_ddim_steps=50,
            guidance_scale=7.5,
            mask_threshold=0.5,
            start_blend=0.2,
            use_replace=False,
            save_img=False
        )

    # 4.真实图片编辑(  Inversion)+ptp+localblend
        img1=run_ptp(
            prompts=["Photo of a dog riding on a little bicycle",
                    f"Photo of a {text} riding on a little bicycle"],
            image_path="./example_images/dog.jpg",
            inv_mode="proxNPI", #NPI为Negative Prompt Inversion，proxNPI为Proximal Inversion
            self_replace_steps = .2, #开始替换自注意力的step
            cross_replace_steps=.3, #开始替换交叉的step
            blend_word = ((('dog',), (f"{text}",))) , #保持bicycle和motorcycle并集以外的部分不被编辑
            eq_params = {"words": (f"{text}",), "values": (2,)},#将motorcycle对应的cross atten map*1
            seed=12345,
            num_ddim_steps=50, # 采样次数
            guidance_scale=7.5, # 去噪时的CFG(加噪时的CFG为1)
            mask_threshold=0.5,  # localblend确定mask时使用的阈值
            start_blend=0.2, # 从20%的step开始使用localblend
            use_replace=False, # 使用refine,而非replace,
            save_img=False
        )
        grid_dict["y_text_list"]+=[f"dog->{text}"]
        if i==0:
            ans+=[img[0],img1[0]]
        ans+=[img[1],img1[1]]

    view_images(ans,num_rows=len(grid_dict["y_text_list"]),text="cmp",grid_dict=grid_dict)