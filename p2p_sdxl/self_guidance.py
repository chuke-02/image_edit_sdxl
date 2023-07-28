from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import torch.nn as nn
from p2p_sdxl.test_utils import LocalBlend
from test_utils import AttentionControlEdit,NUM_DDIM_STEPS
import ptp_utils
from collections import Counter
class SelfGuidanceCompute:
    def __init__(self, prompts: List[str],edit_words:str,tokenizer,max_num_words=77,mode=None,value=None):
        self.batch_size=len(prompts)
        self.tokenizer=tokenizer
        self.max_num_words=max_num_words
        self.tensor_device=self.tokenizer.device
        self.alpha_layers,self.other_layers=self.set_alpha_layers(prompts[0],edit_words)
        self.mode=mode.strip()
        self.value=value

        
    def set_alpha_layers(self,prompt:str,edit_words:str):
        equalizer = torch.zeros(self.max_num_words)
        edit_words=edit_words.split(' ')
        for word in edit_words:
            inds = ptp_utils.get_word_inds(prompt, word, self.tokenizer)
            equalizer[inds] =1
        mask=torch.nonzero(equalizer).squeeze().to(self.tensor_device),
        other_mask=torch.nonzero(equalizer==0).squeeze().to(self.tensor_device),
        return mask,other_mask
    
    def get_maps(self,attention_store,use_pool=False,up_sample=False):# self_attn
        maps = attention_store["down_cross"][10:] + attention_store["up_cross"][:10] 
        maps = [item.reshape(self.batch_size, -1, 1, 32, 32, self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)#[2,40,1,32,32,77]
        #maps=(maps * self.alpha_layers).sum(-1).mean(1)
        maps=maps.mean(1).squeeze(1)#2,32,32,77
        if use_pool:
            maps = nnf.max_pool2d(maps, (3,3), (1, 1), padding=(1, 1))
        if up_sample: 
            maps = nnf.interpolate(maps, size=(128,128))
        self.target_maps,self.other_maps=maps[:,:,:,self.alpha_layers],maps[:,:,:,self.other_layers]
        #2,32,32,?
    def get_activations():
        pass
    
    def compute_shape(self,threshold=0.5):
        shape_target=self.target_maps.gt(threshold)
        shape_other=self.other_maps.gt(threshold)
        return shape_target,shape_other
    
    def compute_centroid(self):
        H,W=self.target_maps.shape[1:3]
        H_cnt=torch.arange(0,H)
        W_cnt=torch.arange(0,W)
        H_centroid_target=torch.einsum("bhwt,h->bt")
        W_centroid_target=torch.einsum("bhwt,w->bt")
        
        H,W=self.other_maps.shape[1:3]
        H_cnt=torch.arange(0,H)
        W_cnt=torch.arange(0,W)
        H_centroid_other=torch.einsum("bhwt,h->bt")
        W_centroid_other=torch.einsum("bhwt,w->bt")
        return H_centroid_target,W_centroid_target,H_centroid_other,W_centroid_other
    
    def compute_size(self):
        H,W=self.target_maps.shape[1:3]
        size_target=torch.sum(self.target_maps,dim=(1, 2))
        H,W=self.other_maps.shape[1:3]
        size_other=torch.sum(self.other_maps,dim=(1, 2))
        return size_target,size_other

    def compute_appearance(self,attention_store,activations):
        maps=self.get_maps(attention_store).squeeze()
        activations=self.get_activations()
        appearance=None
        
        return 0,0

    def exec_edit(self,shape,size,H_centroid,W_centroid,appearance):#shape:32*32*?
        H,W,C=shape.shape
        device=self.tensor_device
        value=self.value
        if self.mode=="up":
            shift_len=int(H*value)
            H_centroid=H_centroid-shift_len
            shape=torch.cat([shape[:,shift_len:,:],torch.new_zeros(shift_len,W,C)],dim=1)
        elif self.mode=="down":
            shift_len=int(H*value)
            H_centroid=H_centroid+shift_len
            shape=torch.cat([torch.new_zeros(shift_len,W,C),shape[:,:shift_len,:]],dim=1)
        elif self.mode=="left":
            shift_len=int(W*value)
            W_centroid=W_centroid-shift_len
            shape=torch.cat([shape[:,:,shift_len:],torch.new_zeros(H,shift_len,C)],dim=2)
        elif self.mode=="right":
            shift_len=int(W*value)
            W_centroid=W_centroid+shift_len
            shape=torch.cat([torch.new_zeros(H,shift_len,C),shape[:,:,:shift_len]],dim=1)
        elif self.mode=="enlarge":
            pass
        elif self.mode=="shrink":
            pass  
        elif self.mode=="width":
            pass
        elif self.mode=="height":
            pass  
        elif self.mode=="swap":
            pass    
        elif self.mode=="restyle":
            pass 
        return shape.to(device),size,H_centroid,W_centroid,appearance
    
    def compute_loss_single(self,shape,size,H_centroid,W_centroid,appearance,compute_shape_loss=True,
                            compute_size_loss=True,compute_centroid_loss=True,compute_appearance_loss=True,need_edit=False) -> Dict:
        shape_ori,shape_new=shape[0],shape[1]
        size_ori,size_new=size[0],size[1]
        H_centroid_ori,H_centroid_new=H_centroid[0],H_centroid[1]
        W_centroid_ori,W_centroid_new=W_centroid[0],W_centroid[1]
        appearance_ori,appearance_new=appearance[0],appearance[1]
        if need_edit:
            shape_new,size_new,H_centroid_new,W_centroid_new,appearance_new=self.exec_edit(shape_new,size_new,H_centroid_new,W_centroid_new,appearance_new)
        loss={}
        if compute_shape_loss:
            loss['shape_loss']=nn.MSELoss(shape_ori,shape_new,reduction='mean')
        if compute_size_loss:
            loss['size_loss']=nn.MSELoss(size_ori,size_new,reduction='mean')
        if compute_centroid_loss:
            loss['centroid_loss']=nn.MSELoss(H_centroid_ori,H_centroid_new,reduction='mean') 
            loss['centroid_loss']+=nn.MSELoss(W_centroid_ori,W_centroid_new,reduction='mean') 
        if compute_appearance_loss:
            loss['appearance_loss']=nn.MSELoss(appearance_ori,appearance_new,reduction='mean')
        return loss
    
    def compute_loss(self,first_as_target=True,centroid=None,size=None,shape=None,appearance=None):
        assert first_as_target or centroid or size or shape or appearance ,"first_as_target为False时后面的属性不能全为None"
        if first_as_target:
            shape_target,shape_other=self.compute_shape()
            H_centroid_target,W_centroid_target,H_centroid_other,W_centroid_other=self.compute_centroid()
            size_target,size_other=self.compute_size()
            appearance_target,appearance_other=self.compute_appearance()
            loss_target=self.compute_loss_single(shape_target,
                                          size_target,
                                          H_centroid_target,
                                          W_centroid_target,
                                          appearance_target,
                                          compute_shape_loss=True,
                                          compute_size_loss=True,
                                          compute_centroid_loss=True,
                                          compute_appearance_loss=False,
                                          need_edit=True
                                          )
            #loss_sum=sum(loss.values())
            loss_other=self.compute_loss_single(shape_other,
                                          size_other,
                                          H_centroid_other,
                                          W_centroid_other,
                                          appearance_other,
                                          compute_shape_loss=True,
                                          compute_size_loss=True,
                                          compute_centroid_loss=True,
                                          compute_appearance_loss=False,
                                          need_edit=False
                                          )
            loss=Counter(loss_target)+Counter(loss_other)
            loss_sum=sum(loss.values())
        return loss_sum,loss
    
    
    def __call__(self,attention_store,activations,latents,first_as_target=True,centroid=None,size=None,shape=None,appearance=None):
        self.get_maps(attention_store)
        self.get_activations(activations)
        loss=self.compute_loss(first_as_target,centroid,size,shape,appearance)
        grad = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
        return grad

class SelfGuidanceEdit(AttentionControlEdit):
    def self_guidance_callback(self,attention_store,activations,index,latents,scheduler):
        if self.self_guidance_compute is not None:
            grad=self.self_guidance_compute(attention_store,activations,latents)
            latents = latents - grad * scheduler.sigmas[index] ** 2
        return latents
        
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float | Tuple[float, float] | Dict[str, Tuple[float, float]], self_replace_steps: float | Tuple[float, float], local_blend: LocalBlend | None,self_guidance_compute:SelfGuidanceCompute|None,self_guidance_weight=None):
        super().__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.self_guidance_compute=self_guidance_compute
        self.self_guidance_weight=self_guidance_weight
        
def make_self_guidance_controller(prompts: List[str], edit_words=None,tokenizer=None) -> AttentionControlEdit:
    if edit_words is None:
        self_guidance_compute=SelfGuidanceCompute(prompts,edit_words,tokenizer)
    else:
        self_guidance_compute=None
    controller=SelfGuidanceEdit(prompts, NUM_DDIM_STEPS, cross_replace_steps=.0,
                                      self_replace_steps=.0, local_blend=None,self_guidance_compute=self_guidance_compute)
    return controller