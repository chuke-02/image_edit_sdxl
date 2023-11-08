from utils.run_code import run_ptp,init_model
from utils.ptp_utils import view_images
prompts=["a cat is standing on the sandbeach",
            "a dog is jumping on the sandbeach"]
prompts0=["a cat is standing on the sandbeach",
            "a cat is jumping on the sandbeach"]
text_list=["motorcycle","car","train","plane","ship"]
prompt=prompts[0]+"\n"+prompts[1]
branch=3
if branch==1:
    grid_dict={"x_title":"start_step:","y_title":"","font_size":100}
    grid_dict["x_text_list"]=['  origin']
    grid_dict["y_text_list"]=[]
    grid_dict['title']=prompt+'\nstep:0...49'
    grid_dict['shift_y']=-300,0
    masa_start_list=[4,10,20,30,40,50]
    start_layer=[45,55,65]
    start_layer_=start_layer[0]
    ans1=[]
    ans2=[]
    ans=[]
    ldm_stable,inversion=init_model(model_path="/stable-diffusion-xl-base-1.0")
    for start_layer_ in start_layer:
        ans1=[]
        ans2=[]
        grid_dict["y_text_list"]+=[f'masa w/o mask\nstart_layer:{start_layer_}',f'masa w/ mask\nstart_layer:{start_layer_}']
        for i in masa_start_list:

            # img=run_ptp(
            #     prompts=prompts,
            #     #image_path="./example_images/cat_bike3.jpg",
            #     #inv_mode="NPI",
            #     self_replace_steps = 0.0,
            #     cross_replace_steps=0.0,
            #     blend_word = ((('cat',), ("cat",))) ,
            #     eq_params = None,
            #     seed=2,
            #     num_ddim_steps=50,
            #     guidance_scale=7.5,
            #     mask_threshold=0.5,
            #     start_blend=0.2,
            #     use_replace=False,
            #     model_path="/stable-diffusion-xl-base-1.0",
            #     masa_control=False,
            #     x_t_replace=False,
            #     save_img=False,
            #     masa_start=i
            # )
            if start_layer_==start_layer[0]:
                grid_dict["x_text_list"]+=[str(i)] if i <50 else [str(i)+"(No masa)"]
            img1=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.2,
                cross_replace_steps=0.3,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img2=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.2,
                cross_replace_steps=0.3,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=True,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,#True的话启用localblend，False的话不用localblend
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            if i==masa_start_list[0]:
                ans1.append(img1[0])
            ans1.append(img1[1])
            if i==masa_start_list[0]:
                ans2.append(img2[0])
            ans2.append(img2[1])
        ans+=ans1+ans2
    view_images(ans,num_rows=6,text="masa_cmp",grid_dict=grid_dict)
# run_ptp(
#     prompts=["Photo of a cat riding on a little bicycle",
#             "Photo of a cat flying on a little bicycle"],
#     image_path="./example_images/cat_bike3.jpg",
#     inv_mode="NPI",
#     self_replace_steps = 1.0,
#     cross_replace_steps=1.0,
#     blend_word = ((('cat',), ("cat",))) ,
#     eq_params = None,
#     seed=1234567,
#     num_ddim_steps=50,
#     guidance_scale=7.5,
#     mask_threshold=0.4,
#     start_blend=0.2,
#     use_replace=False,
#     model_path="/stable-diffusion-xl-base-1.0",
#     masa_control=False,
#     x_t_replace=False
# )
elif branch==2:
    grid_dict={"x_title":"","y_title":"","font_size":100}
    grid_dict["x_text_list"]=['  origin',"ptp","masa*","masa","masa\n+ptp(cross_attn)","masa\n+ptp","None"]
    grid_dict["y_text_list"]=['masa w/o mask\nstart_layer:55\nstart_step:10']
    grid_dict['title']=prompt+'\n*:'+prompts0[1]
    grid_dict['shift_y']=-300,0
    masa_start_list=[10]
    start_layer=[55]
    start_layer_=start_layer[0]
    ans1=[]
    ans2=[]
    ans=[]
    ldm_stable,inversion=init_model(model_path="/stable-diffusion-xl-base-1.0")
    for start_layer_ in start_layer:
        ans1=[]
        ans2=[]
        #grid_dict["y_text_list"]+=[f'masa w/o mask\nstart_layer:{start_layer_}',f'masa w/ mask\nstart_layer:{start_layer_}']
        for i in masa_start_list:

            # img=run_ptp(
            #     prompts=prompts,
            #     #image_path="./example_images/cat_bike3.jpg",
            #     #inv_mode="NPI",
            #     self_replace_steps = 0.0,
            #     cross_replace_steps=0.0,
            #     blend_word = ((('cat',), ("cat",))) ,
            #     eq_params = None,
            #     seed=2,
            #     num_ddim_steps=50,
            #     guidance_scale=7.5,
            #     mask_threshold=0.5,
            #     start_blend=0.2,
            #     use_replace=False,
            #     model_path="/stable-diffusion-xl-base-1.0",
            #     masa_control=False,
            #     x_t_replace=False,
            #     save_img=False,
            #     masa_start=i
            # )
            img1=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.0,
                cross_replace_steps=0.0,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=False,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img2=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.2,
                cross_replace_steps=0.3,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=False,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img3=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.0,
                cross_replace_steps=0.0,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img3_1=run_ptp(
                prompts=prompts0,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.0,
                cross_replace_steps=0.0,
                blend_word = ((('cat',), ("cat",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img4=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.0,
                cross_replace_steps=0.3,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
            img5=run_ptp(
                prompts=prompts,
                #image_path="./example_images/cat_bike3.jpg",
                #inv_mode="NPI",
                self_replace_steps = 0.2,
                cross_replace_steps=0.3,
                blend_word = ((('cat',), ("dog",))) ,
                eq_params = None,
                seed=1234567,
                num_ddim_steps=50,
                guidance_scale=7.5,
                mask_threshold=0.5,
                start_blend=0.2,
                use_replace=False,
                model_path="/stable-diffusion-xl-base-1.0",
                masa_control=True,
                masa_mask=False,
                masa_start_step=i,
                masa_start_layer=start_layer_,
                x_t_replace=False,
                save_img=False,
                model=ldm_stable,
                inversion=inversion
            )
    view_images([img1[0]]+[img2[1]]+[img3_1[1]]+[img3[1]]+[img4[1]]+[img5[1]]+[img1[1]],num_rows=1,text="masa_cmp",grid_dict=grid_dict)

elif branch==3:
    prompts=["a bird",
            "a bird is spreading wings"]
    ldm_stable,inversion=init_model(model_path="/stable-diffusion-xl-base-1.0")
    run_ptp(
        prompts=prompts,
        #image_path="./example_images/cat_bike3.jpg",
        #inv_mode="NPI",
        self_replace_steps = 0.0,
        cross_replace_steps=0.0,
        blend_word = ((('bird',), ("bird",))) ,
        eq_params = None,
        seed=123,
        num_ddim_steps=50,
        guidance_scale=7.5,
        mask_threshold=0.5,
        start_blend=0.2,
        use_replace=False,
        model_path="/stable-diffusion-xl-base-1.0",
        masa_control=True,
        masa_mask=False,
        masa_start_step=10,
        masa_start_layer=55,
        x_t_replace=False,
        save_img=True,
        model=ldm_stable,
        inversion=inversion
        )