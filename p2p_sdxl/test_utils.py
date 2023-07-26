import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from torchvision import utils
from torch.optim.adam import Adam
from PIL import Image
from sdxl import sdxl
import matplotlib.pyplot as plt
import seaborn as sns
pipe = sdxl.from_pretrained("/home/cas/stable-diffusion-xl-base-0.9", torch_dtype=torch.bfloat16, use_safetensors=True, variant="bf16")
pipe.to("cuda")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe.scheduler=scheduler #用DDIM scheduler

MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
NUM_INNER_STEPS=30 #null text Inversion中每个step优化几次
FP16=False
HEATMAP=True #是否可视化heatmap
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable=pipe
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer
class LocalBlend:

    def get_mask(self, maps, alpha, use_pool,x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        if HEATMAP and self.counter%10==0:
            sns.heatmap(mask[0][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap0_old_{self.counter}.png')
            plt.clf()
            sns.heatmap(mask[1][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap1_old_{self.counter}.png')
            plt.clf()
        mask=(mask - mask.min ()) / (mask.max () - mask.min ())
        if HEATMAP and self.counter%10==0:
            # utils.save_image(mask[0].cpu()*1.0, './vis/image1.png',cmap='gray')
            # utils.save_image(mask[1].cpu()*1.0, './vis/image2.png',cmap='gray')
            # for i in range(10):
            #     utils.save_image(mask.gt(0.1*i)[0].cpu()*1.0, f"./vis/image1gt{i}.png",cmap='gray')
            #     utils.save_image(mask.gt(0.1*i)[1].cpu()*1.0, f"./vis/image2gt{i}.png",cmap='gray')
            
            sns.heatmap(mask[0][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap0_{self.counter}.png')
            plt.clf()
            sns.heatmap(mask[1][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap1_{self.counter}.png')
            plt.clf()
            utils.save_image((mask.gt(0.4)[0]+mask.gt(0.4)[1]).cpu()*1.0, f"./vis/image_mask.png",cmap='gray')
        mask = mask.gt(0.6)

        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            # 40,1024,77
            maps = attention_store["down_cross"][10:] + attention_store["up_cross"][:10]#20个down,10个mid,30个up,v1.4中为4,1,6
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 32, 32, MAX_NUM_WORDS) for item in maps]#原来是16,16.似乎该操作有点慢
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True,x_t)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float().to(x_t.dtype)
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words, substruct_words=None, start_blend=0.2,
                 th=(.6, .6)):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th = th


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
             x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 64 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        self.mapper=self.mapper.to(pipe.unet.dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def aggregate_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb,
                                       controller=controller)
    return controller


def show_cross_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image

class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        print(f"prev_step:timestep {timestep} prev {prev_timestep}")
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        print(f"next_step:timestep {timestep} next {next_timestep}")
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context,cond=True,both=False):
        added_cond_id=1 if cond else 0
            
        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance=False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        if both is False:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds[added_cond_id].unsqueeze(0), "time_ids": self.add_time_ids[added_cond_id].unsqueeze(0)}
        else:
            added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=context,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]#1,4,128,128
        #old:noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred=self.get_noise_pred_single(latents_input,t,context,both=True)
        # old:noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents) #x_0 ->x_t,加噪
        else:
            latents = self.prev_step(noise_pred, t, latents) #x_t ->x_0，生图
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.model.vae.config.scaling_factor * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None):
        original_size = original_size or (1024, 1024)
        target_size = target_size or (1024, 1024)
        # 3. Encode input prompt
        do_classifier_free_guidance=True
        (
            prompt_embeds,# 2,77,2048
            negative_prompt_embeds,# 2,77,2048
            pooled_prompt_embeds,# 2,1280
            negative_pooled_prompt_embeds,# 2,1280
        ) = self.model.encode_prompt_not_zero_uncond(
            prompt,
            self.model.device,
            1,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )
        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.model._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        self.add_text_embeds = add_text_embeds.to(device)
        self.add_time_ids = add_time_ids.to(device).repeat(1 * 1, 1)

        self.prompt_embeds=prompt_embeds
        self.negative_prompt_embeds=negative_prompt_embeds
        self.pooled_prompt_embeds=pooled_prompt_embeds
        self.negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
        self.prompt = prompt
        self.context=prompt_embeds
        # old:uncond_input = self.model.tokenizer(
        #     [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
        #     return_tensors="pt"
        # )
        # uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        # text_input = self.model.tokenizer(
        #     [prompt],
        #     padding="max_length",
        #     max_length=self.model.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        # self.context = torch.cat([uncond_embeddings, text_embeddings])#2*77*768
        # self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            #print(i)
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings,cond=True)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image) #img:# 512,512,3          latent:1,4,64,64
        image_rec = self.latent2image(latent) # 512,512,3
        ddim_latents = self.ddim_loop(latent.to(self.model.unet.dtype)) # [1,4,64,64]*steps
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=0.1 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings,cond=True) #1,4,64,64
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings,cond=False)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=NUM_INNER_STEPS, early_stop_epsilon=1e-5,
               verbose=False,train_free=True):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_1024(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        #print(1)
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if train_free is False:
            if verbose:
                print("Null-text optimization...")
            #st=ddim_latents[-1]
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
            #w=st-ddim_latents[-1]
            return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        else:
            if verbose:
                print("Negative-prompt Inversion...")
            return (image_gt, image_rec), ddim_latents[-1], self.prompt_embeds[1].unsqueeze(0),self.pooled_prompt_embeds


    def __init__(self, model):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.model.vae.to(dtype=torch.float32)
        self.prompt = None
        self.context = None


null_inversion = NullInversion(ldm_stable)


@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        height=None,
        width=None,
        null_inversion=False,
        pooled_uncond_embeddings=None
):
    return model(controller=controller,
                 prompt=prompt,
                 latents=latent,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=generator,
                 negative_prompt_embeds=uncond_embeddings,
                 negative_pooled_prompt_embeds=pooled_uncond_embeddings,
                 p2p=True,
                 height=height,
                 width=width,
                 same_init=True,
                 null_inversion=null_inversion
                 )


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                    verbose=True,use_old=False,one_img=False,null_inversion=False,text="",pooled_uncond_embeddings=None,folder=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                        num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                        generator=generator, uncond_embeddings=uncond_embeddings,null_inversion=null_inversion,pooled_uncond_embeddings=pooled_uncond_embeddings)
    images=np.asarray(images)
    if verbose:
        if use_old:
            ptp_utils.view_images_old(images)
        else:
            ptp_utils.view_images(images,Notimestamp=one_img,text=text,folder=folder)
    return images, x_t

def init_model():
    pass
# def init_model(ddim_scheduler=True,):
#     pipe = sdxl.from_pretrained("/home/cas/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
#     pipe.to("cuda")
#     if ddim_scheduler:
#         scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
#         pipe.scheduler=scheduler

#     NUM_DDIM_STEPS = 50
#     GUIDANCE_SCALE = 7.5
#     MAX_NUM_WORDS = 77
#     FP16=True
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     ldm_stable=pipe
#     try:
#         ldm_stable.disable_xformers_memory_efficient_attention()
#     except AttributeError:
#         print("Attribute disable_xformers_memory_efficient_attention() is missing")
#     tokenizer = ldm_stable.tokenizer