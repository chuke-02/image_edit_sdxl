# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import torch.nn.functional as F

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


from datetime import datetime


def view_images(images, num_rows=1, offset_ratio=0.02,text="",folder=None,Notimestamp=False,grid_dict=None,subfolder=None,verbose=True):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    for i, per_image in enumerate(images):
        if isinstance(per_image, Image.Image):
            images[i] = np.array(per_image)
        
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)

    pil_img_=draw_axis(pil_img,grid_dict,num_cols,num_rows)
    if pil_img_.size[0]==pil_img_.size[1]:
        pil_img_.resize((2048,2048))
    else:
        longer_side = max(pil_img.size)
        ratio = 2048/longer_side
        new_size = tuple([int(x*ratio) for x in pil_img.size])
        pil_img = pil_img.resize(new_size)
    now = datetime.now()
    if Notimestamp is False:
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp=""
    if folder is not None:
        dirname="./"+folder
        filename = text+f"img_{timestamp}.jpg"
    else:
        if subfolder is not None:
            dirname=os.path.join("./img", subfolder,now.strftime("%Y-%m-%d"))
            dirname=os.path.join(dirname,now.strftime("%H-%M-%S"))            
            filename =text+f"img_{timestamp}.jpg"
        else:
            dirname=os.path.join("./img",now.strftime("%Y-%m-%d"))
            dirname=os.path.join(dirname,now.strftime("%H-%M-%S"))
            filename =text+f"img_{timestamp}.jpg"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if verbose is True:
        for i, img in enumerate(images):
            im = Image.fromarray(img)
            im.save(os.path.join(dirname,f"{i}.jpg"))
    print(f"Output dir: {dirname}")
    pil_img.save(os.path.join(dirname, filename))
    #dirname=os.path.join(dirname, "2048x")
    if grid_dict is not None and grid_dict is not False:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pil_img_.save(os.path.join(dirname, filename[:-4]+"_2048x.jpg"))
    

def draw_axis(img,grid_dict,x_len,y_len):
    if grid_dict is not None and grid_dict is not False:
        assert isinstance(grid_dict,Dict)
        assert "x_title" in grid_dict
        assert "y_title" in grid_dict
        assert "x_text_list" in grid_dict
        assert "y_text_list" in grid_dict
        x_title=grid_dict["x_title"]
        y_title=grid_dict["y_title"]
        x_text_list=grid_dict['x_text_list']
        y_text_list=grid_dict['y_text_list']
        assert len(y_text_list)==y_len
        assert len(x_text_list)==x_len
        assert "font_size" in grid_dict
        font_size=grid_dict["font_size"]
        if "x_color" in grid_dict:
            color_x=grid_dict['x_color']
        else:
            color_x="black"
        if "y_color" in grid_dict:
            color_y=grid_dict['y_color']
        else:
            color_y="black"
        if "num_decimals" in grid_dict:
            num_decimals=grid_dict['num_decimals']
        else:
            num_decimals=2
        if "shift_x" in grid_dict:
            shift_x_x,shift_x_y=grid_dict['shift_x']
        else:
            shift_x_x=shift_x_y=0
        if "shift_y" in grid_dict:
            shift_y_x,shift_y_y=grid_dict['shift_y']
        else:
            shift_y_x=shift_y_y=0
        if "title" in grid_dict:
            title=grid_dict['title']
            if isinstance(title,List):
                all_title=""
                for s in title:
                    all_title=all_title+s+"\n"
                title=all_title
        else:
            title=''
        width, height = img.size
        num_x=x_len
        num_y=y_len
        #color_x="black"
        #color_y="black"
        
        new_img = Image.new("RGB", (width + width // num_x+width // (num_x*2), height + height // num_y+height // (num_y*2)), color=(255, 255, 255))
        width,height=(width + width // num_x, height + height // num_y)
        num_x=num_x+1
        num_y=num_y+1
        new_img.paste(img, (width // num_x, height // num_y))

        draw = ImageDraw.Draw(new_img)

        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        for i in range(2, num_x+1):
            x = (i - 1) * width // num_x + width // (num_x * 2)-width *0.2// num_x+shift_x_x
            y = height // (num_y * 2)+shift_x_y
            k=i-1
            if  isinstance(x_text_list[i-2],str):
                draw.text((x, y), x_text_list[i-2], font=font,fill=color_x,align="center")
            else:
                draw.text((x, y), "{:.{}f}".format(x_text_list[i-2],num_decimals), font=font,fill=color_x,align="center")

        for i in range(2, num_y+1):
            x = width // (num_x * 2)-width *0.1// num_x+shift_y_x
            y = (i - 1) * height // num_y + height // (num_y * 2)-height*0.1//num_y+shift_y_y
            k = i - 1
            if isinstance(y_text_list[i-2],str):
                draw.text((x, y), y_text_list[i-2], font=font,fill=color_y,align="center")
            else:
                draw.text((x, y), "{:.{}f}".format(y_text_list[i-2],num_decimals), font=font,fill=color_y,align="center")
        i=1
        x = (i - 1) * width // num_x + width // (num_x * 2)-height*0.1//num_y+shift_y_x
        y = height // (num_y * 2)+width *0.2// num_x+shift_y_y
        draw.text((x, y), y_title, font=font, fill=color_y,align="center")
        x = width // (num_x * 2)+width *0.2// num_x+shift_x_x
        y = (i - 1) * height // num_y + height // (num_y * 2)+shift_x_y
        draw.text((x, y), x_title, font=font, fill=color_x,align="left")
        x = width // 4
        y = (i - 1) * height // num_y + height // (num_y * 10)
        draw.text((x, y), title, font=font, fill='blue',align="left")
        # ret_img = Image.new("RGB", (width + width // (num_x*2),
        #                              height + height // (num_y*2)), color=(255, 255, 255))
        # ret_img.paste(new_img, (0,0))
        # new_img=ret_img
    else:
        #print("grid_dict格式错误,跳过图例的生成")
        new_img=img
    return new_img

def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    extra_set_kwargs = {}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    model.scheduler.config.steps_offset = 1
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller,masa_control=False,masa_mask=True,masa_start_step=40,masa_start_layer=55):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape #当输入两个prompt时,batch_size=4,即:原prompt_uncond,新prompt_uncond,原prompt_cond,新prompt_cond
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)#10,4096,64      10,4096,64
            k = self.head_to_batch_dim(k)#10,4096,64      10,77,64
            v = self.head_to_batch_dim(v)#10,4096,64      10,77,64
            if hasattr(controller, 'count_layers'): #给masa controll用的
                controller.count_layers(place_in_unet,is_cross)
            if masa_control is True and is_cross is False:#给masa controll用的
                #qkv={"q":q,"k":k,"v":v}
                q,k,v,masa_control_mask=controller.replace_self_attention_kv(q,k,v,h,place_in_unet,masa_start_step,masa_start_layer) 
                if masa_mask is True and masa_control_mask is not None:
                    sim_fg = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                    sim_bg = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                    size=int(np.sqrt(q.shape[1]))
                    masa_control_mask=F.interpolate(masa_control_mask.to(q.dtype),(size,size))

                    new_prompt_batch_size=batch_size//2-1
                    #mask_tar=F.interpolate(mask_tar,(size,size))
                    mask_src,mask_tar=masa_control_mask[0],masa_control_mask[1:]
                    mask_src = mask_src.expand(new_prompt_batch_size, -1,-1)
                    mask_src = mask_src.reshape(new_prompt_batch_size, -1)
                    mask_src = mask_src[:, None, :].repeat(h, 1, 1)
                    mask_src_one=torch.ones_like(mask_src[:h])
                    mask_src=torch.cat([mask_src_one,mask_src]*2,dim=0) 
                    mask_src=mask_src.gt(0.5) #转换回bool

                    mask_tar=mask_tar.squeeze(1).reshape(new_prompt_batch_size, -1)
                    mask_tar = mask_tar.reshape(new_prompt_batch_size, -1)
                    mask_tar = mask_tar[:, :,None].repeat(h, 1,1)
                    mask_tar_one=torch.ones_like(mask_tar[:h])
                    mask_tar=torch.cat([mask_tar_one,mask_tar]*2,dim=0)
                    #mask_tar=mask_tar.gt(0.5) #转换回bool

                    max_neg_value = -torch.finfo(sim_fg.dtype).max
                    
                    sim_fg.masked_fill_(~mask_src, max_neg_value).softmax(dim=-1)
                    sim_bg.masked_fill_(mask_src, max_neg_value).softmax(dim=-1)
                    sim_fg=controller(sim_fg, is_cross, place_in_unet)
                    sim_bg=controller(sim_bg, is_cross, place_in_unet)
                    controller.cur_att_layer=controller.cur_att_layer-1
                    attn = torch.cat([sim_fg, sim_bg])

                    #attn = sim.softmax(dim=-1)
                    if len(attn) == 2 * len(v):
                        v = torch.cat([v] * 2)
                    out = torch.einsum("b i j, b j d -> b i d", attn, v)
                    out_fg,out_bg=out.chunk(2) #[40,4096,64]
                    #torch.einsum("b i d, b i -> b i d", out_fg, mask_tar)
                    out=out_fg*mask_tar+out_bg*(1-mask_tar)
                    out = self.batch_to_head_dim(out)
                    return to_out(out)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of

            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)#10,4096,4096   10,4096,77
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

def view_images_old(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)