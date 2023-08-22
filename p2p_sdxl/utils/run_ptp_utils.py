import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler,DDIMInverseScheduler,DPMSolverMultistepInverseScheduler,EulerDiscreteScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import utils.ptp_utils as ptp_utils

import utils.seq_aligner as seq_aligner
from torchvision import utils
from torch.optim.adam import Adam
from PIL import Image
from utils.sdxl import sdxl
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ddim_scheduler import DDIMSchedulerDev
from utils.inversion import load_1024_mask,Inversion
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
HEATMAP=False #是否可视化heatmap
MASK_FILE=False

def get_mask_file():
    mask=load_1024_mask("example_images/dog_dog_mask.jpg")
    mask=mask//255
    mask=torch.from_numpy(mask).unsqueeze(0)
    return mask
class LocalBlend:
    def get_mask(self, maps, alpha, use_pool,x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask=(mask - mask.min ()) / (mask.max () - mask.min ())
        if HEATMAP and self.counter%10==0:#从某一个step开始
            sns.heatmap(mask[0][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap0_{self.counter}.png')
            plt.clf()
            sns.heatmap(mask[1][0].clone().cpu(), cmap='coolwarm')
            plt.savefig(f'./vis/heatmap1_{self.counter}.png')
            plt.clf()
            utils.save_image((mask.gt(self.mask_threshold)[0]+mask.gt(self.mask_threshold)[1]).cpu()*1.0, f"./vis/image_mask.png",cmap='gray')
        mask = mask.gt(self.mask_threshold)
        self.mask=mask
        mask = mask[:1] + mask
        return mask


    def __call__(self, x_t, attention_store):
        self.counter += 1
        self.mask=None
        self.origin_mask=None
        # 40,1024,77
        if attention_store is not None:
            if MASK_FILE is False:
                maps = attention_store["down_cross"][10:] + attention_store["up_cross"][:10]#20个down,10个mid,30个up,v1.4中为4,1,6
                maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 32, 32, MAX_NUM_WORDS) for item in maps]#原来是16,16
                maps = torch.cat(maps, dim=1)
                mask = self.get_mask(maps, self.alpha_layers, True,x_t)
                if self.substruct_layers is not None:
                    maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                    mask = mask * maps_sub
                mask = mask.float().to(x_t.dtype)
            else:
                mask=self.mask_file.expand_as(x_t).to(x_t.device)
                self.mask=mask
        if self.counter > self.start_blend and self.x_t_replace is True:
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words, tokenizer,device,num_ddim_steps,substruct_words=None, start_blend=0.2,
                 mask_threshold=0.6,x_t_replace=True,**kwargs):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        self.tokenizer=tokenizer
        self.mask_threshold=mask_threshold
        self.model_device=device
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, self.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, self.tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(self.model_device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(self.model_device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0
        self.mask=None
        if MASK_FILE:
            self.mask_file=get_mask_file()
        self.x_t_replace=x_t_replace


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

    def __init__(self, stop_inject: float,num_ddim_steps):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)


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
                    if self.masa_control is False :
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

    def count_layers(self,place_in_unet,is_cross):
        if self.last_status=='up' and place_in_unet=='down':
            self.self_layer=0
            self.cross_layer=0
        self.last_status=place_in_unet
        if is_cross is True:
            self.cross_layer=self.cross_layer+1
        else:
            self.self_layer=self.self_layer+1
        #print(self.self_layer,self.cross_layer)

        
    def replace_self_attention_kv(self,q,k,v,heads,place_in_unet,masa_start_step,masa_start_layer):
        #print(self.local_blend.counter)
        if self.local_blend.counter >= masa_start_step and self.local_blend.mask is not None  and self.self_layer>=masa_start_layer:
            #q,k,v均为[40,4096,64]
            print(masa_start_step)
            kv=torch.cat([k,v],dim=0)
            split_kv = kv.split(heads, dim=0)
            new_kv = torch.stack(split_kv, dim=0)# [8(batch_size*4),10,4096,64]
            split_new_kv = torch.chunk(new_kv, chunks=4, dim=0)# 
            new_kv = torch.stack(split_new_kv, dim=0)# [4,batch_size,10,4096,64] ,第0维的4代表k_u,k_c,v_u,v_c
            batch_size=new_kv.shape[0]
            assert batch_size>1,"masa control : batch_size > 1"
            new_kv[:,1:]=new_kv[:,0].unsqueeze(1).expand_as(new_kv[:,1:])
            new_kv=new_kv.reshape(-1,*new_kv.shape[-2:])
            k,v=new_kv.chunk(2)
            if  False and self.local_blend.counter%10==0 and self.last_save!=self.local_blend.counter:
                sns.heatmap(self.local_blend.mask[0][0].clone().cpu(), cmap='coolwarm')
                plt.savefig(f'./vis/masa_control/mask0_{self.local_blend.counter}.png')
                plt.clf()
                sns.heatmap(self.local_blend.mask[1][0].clone().cpu(), cmap='coolwarm')
                plt.savefig(f'./vis/masa_control/mask1_{self.local_blend.counter}.png')
                plt.clf()
                self.last_save=self.local_blend.counter
            return q,k,v,self.local_blend.mask#[2,1,128,128]
        else:
            return q,k,v,None

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

    def __init__(self, prompts, num_steps: int,model,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],masa_control=False):
        super(AttentionControlEdit, self).__init__()
        self.model_dtype=model.unet.dtype
        self.tokenizer=model.tokenizer
        self.model_device=model.unet.device
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            self.tokenizer).to(self.model_device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.masa_control=masa_control
        self.last_save=None
        self.self_layer=0
        self.cross_layer=0
        self.last_status='up'


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int,model, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,masa_control=False):
        super(AttentionReplace, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps, local_blend,masa_control=masa_control)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(self.model_device)
        self.mapper=self.mapper.to(self.model_dtype)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, model,cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,masa_control=False):
        super(AttentionRefine, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps, local_blend,masa_control=masa_control)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.model_device), alphas.to(self.model_device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, model,cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,masa_control=False):
        super(AttentionReweight, self).__init__(prompts, num_steps,model, cross_replace_steps, self_replace_steps,
                                                local_blend,masa_control=masa_control)
        self.equalizer = equalizer.to(self.model_device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]],model):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, model.tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(prompts: List[str], model,num_ddim_steps,is_replace_controller: bool, cross_replace_steps: Dict[str, float],
                    self_replace_steps: float, blend_words=None, equilizer_params=None,masa_control=False,**kwargs) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words,model.tokenizer,model.unet.device,num_ddim_steps,**kwargs)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, model,cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, local_blend=lb,masa_control=masa_control)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, model,cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, local_blend=lb,masa_control=masa_control)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"],model)
        controller = AttentionReweight(prompts, num_ddim_steps,model, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb,
                                       controller=controller,masa_control=masa_control)
    return controller



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
        pooled_uncond_embeddings=None,
        **kwargs
):
    return model(
        controller=controller,
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
        null_inversion=null_inversion,
        return_dict=False,
        **kwargs
        )


def run_and_display(prompts, controller, ldm_stable,num_ddim_steps,guidance_scale,latent=None, run_baseline=False, generator=None, uncond_embeddings=None,
                    verbose=True,use_old=False,one_img=False,null_inversion=False,text="",pooled_uncond_embeddings=None,folder=None,save_per_img=True,**kwargs):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator,**kwargs)
        print("with prompt-to-prompt")
    images = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                        num_inference_steps=num_ddim_steps, guidance_scale=guidance_scale,
                                        generator=generator, uncond_embeddings=uncond_embeddings,null_inversion=null_inversion,pooled_uncond_embeddings=pooled_uncond_embeddings,**kwargs)[0]
    images=np.array(images)
    if verbose:
        if use_old:
            ptp_utils.view_images_old(images)
        else:
            ptp_utils.view_images(images,Notimestamp=one_img,text=text,folder=folder,verbose=save_per_img)
    return images

