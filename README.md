# 概述
本仓库基于[Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)实现了以下算法：

可控图片生成：

- [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)
- [MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://arxiv.org/abs/2304.08465)

Inversion：
- [Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models](https://arxiv.org/abs/2305.16807)
-  [Improving Tuning-Free Real Image Editing with Proximal Guidance](https://arxiv.org/abs/2306.05414)

本仓库的实现参考了  
https://github.com/google/prompt-to-prompt  
https://github.com/TencentARC/MasaCtrl

# 使用
## 配置环境
使用docker:
```dockerfile
docker pull chuke02/sdxl:20230903
```
使用conda:
```
docker pull chuke02/sdxl:20230903
```
## prompt to prompt 和 masa control

后文对应的代码在example.py中。    
ptp:

```python
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    self_replace_steps = .2,#开始替换自注意力的step
    cross_replace_steps=.3,#开始替换交叉注意力的step
    eq_params = {"words": ("motorcycle",), "values": (1,)},#将motorcycle对应的cross attention乘1
    seed=12345,
    num_ddim_steps=50,
    guidance_scale=7.5,
    use_replace=False,#取False则使用refine，否则使用replace，一般而言refine效果更佳
    model_path="stabilityai/stable-diffusion-xl-base-1.0"
)
```

![img](markdown_images/ptp.jpg)

ptp+localblend:

```python
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
```
![img](markdown_images/ptp_lb.jpg)
真实图片编辑(Proximal Inversion)+ptp+localblend:
```python
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
```
![img](markdown_images/real_img_ptp.jpg)
4.真实图片编辑(Negative Prompt Inversion)+ptp+localblend:
```python
run_ptp(
    prompts=["Photo of a cat riding on a little bicycle",
            "Photo of a cat riding on a little motorcycle"],
    image_path="./example_images/cat_bike3.jpg",
    inv_mode="NPI", #NPI为Negative Prompt Inversion，proxNPI为Proximal Inversion
    self_replace_steps = .2, #开始替换自注意力的step
    cross_replace_steps=.3, #开始替换交叉注意力的step
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
```
![img](markdown_images/real_img_ptp_lb.jpg)
5.masa control(用于进行姿态上的编辑):
```python
run_ptp(
    prompts=["a cat is sitting",
        "a cat is laying"],
    self_replace_steps =0.0, #不使用ptp(可以同时使用ptp，但是效果有点奇怪)
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
    masa_mask=False, # 是否使用基于mask的masa control，如果使用，要设定对应的blend_word,有时候似乎有bug?
    masa_start_step=10, #从第step 10开始进行masa control（替换self attention的 kv） 
    masa_start_layer=45, #从unet的第45个self attention开始替换
    x_t_replace=False, #True的话启用localblend，False的话不用localblend(获取mask但不进行x_t的替换)
)
```
![img](markdown_images/masa_ctrl.jpg)
真实图片编辑+ptp是ok的  
真实图片编辑+masa control也是ok的  
ptp+masa control效果会有点怪  
## 仅inversion
以下介绍如何修改sdxl的pipeline，使其支持inversion  
见example_inversion.py
```python
# 1. 准备模型
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = sdxl.from_pretrained(
    "/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16,
    use_safetensors=True, 
    variant="fp16", 
)
scheduler = DDIMScheduler.from_config(model.scheduler.config)
model.scheduler=scheduler
model.to(device)
num_ddim_steps = 50
inversion = Inversion(model, num_ddim_steps)
prompt="Photo of a cat riding a bike"
inv_mode="proxNPI"
if inv_mode=="proxNPI":
    prox_guidance=True
elif inv_mode=="NPI":
    prox_guidance=False
# 进行inversion
(
    (image_gt, image_enc),  # 分别为原图和inversion效果的上界(用vae encode后立刻decode出来)
    x_T,  # 用于去噪的初始噪声
    x_stars,  # inversion过程中各个step的latent组成的List
    prompt_embeds,  # 原prompt的embdding
    pooled_prompt_embeds # 原prompt的pooled embdding (SDXL有两个embdding)
) = inversion.invert(
    image_path="example_images/cat_bike1.jpg", #图片路径
    prompt=prompt # 原图prompt
)
# 进行infer
image=model(
    prompt=prompt,
    latents=x_T,
    num_inference_steps=num_ddim_steps,
    prox_guidance=prox_guidance, 
    guidance_scale=7.5,
    negative_prompt_embeds=prompt_embeds,
    negative_pooled_prompt_embeds=pooled_prompt_embeds,
    same_init=True,
    x_stars=x_stars,
    return_dict=False
)[0]
view_images([image_gt,image[0]])
```
左边为原图，右边为重建后的图
![img](markdown_images/reconstrction.jpg)
此外，使用inversion需要在infer的过程中做一些操作，见utils/sdxl_inversion.py中sdxl的__call__方法，以StableDiffusionXLPipeline(diffusers==0.18.2)的__call__为基准，我在增加或修改的代码处做了# ADD 或 # CHANGE的标记，具体内容如下：
添加了三个输入
```python
def __call__(
    self,
    ......
    same_init=False, # ADD ，各个prompt表示是否以同一高斯噪声为起点
    x_stars=None, # ADD ，用于porx inversion
    prox_guidance=False, # ADD ，为False时为negative prompt inversion，反之为porx inversion
    ):
```
为了使代码可以在 diffusers==0.20.2下运行，修改了
```python
# CHANGE START
self.check_inputs(
    prompt,
    None,
    height,
    width,
    callback_steps,
    negative_prompt,
    None,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
)
# CHANGE END
```
实现了same_init
```python
latents = self.prepare_latents(
    batch_size * num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    prompt_embeds.dtype,
    device,
    generator,
    latents,
    same_init=same_init #ADD
)
```
在porx inversion中对noise_con和noise_uncond的差做了正则
```python
if do_classifier_free_guidance:
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    # CHANGE START
    score_delta,mask_edit=self.prox_regularization(
        noise_pred_uncond,
        noise_pred_text,
        i,
        t,
        prox_guidance=prox_guidance,
    )
    noise_pred = noise_pred_uncond + guidance_scale * score_delta
    # CHANGE END
```
实现了porx inversion中的porx guidance
```python
# ADD START
latents = self.proximal_guidance(
    i,
    t,
    latents,
    mask_edit,
    prox_guidance=prox_guidance,
    dtype=self.unet.dtype,
    x_stars=x_stars
)
# ADD END
```