from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
class SelfGuidanceBase:
    def __init__(self, prompts: List[str],max_num_words=77):
        self.batch_size=len(prompts)
        self.max_num_words=max_num_words
        self.alpha_layers=self.set_alpha_layers()
        self.other_layers=~self.alpha_layers
    
    def set_alpha_layers(self):
        alpha_layers=torch.zeros(self.batch_size, 1, 1, 1, 1, self.max_num_words)
        pass
    
    def get_maps(self,attention_store,use_pool=False,up_sample=False):# self_attn
        maps = attention_store["down_cross"][10:] + attention_store["up_cross"][:10] 
        maps = [item.reshape(self.batch_size, -1, 1, 32, 32, self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)#[2,40,1,32,32,77]
        maps=(maps * self.alpha_layers).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (3,3), (1, 1), padding=(1, 1))
        if up_sample: 
            maps = nnf.interpolate(maps, size=(128,128))
        return maps
    
    def get_activations():
        pass
    
    def compute_shape(self,attention_store,threshold=0.5):
        maps=self.get_maps(attention_store)
        return maps.gt(threshold)
    
    def compute_centroid(self,attention_store):
        maps=self.get_maps(attention_store).squeeze()
        H,W=maps.shape[1:]
        H_cnt=torch.arange(0,H)
        W_cnt=torch.arange(0,W)
        H_centroid=torch.einsum("bhw,h->b")
        W_centroid=torch.einsum("bhw,w->b")
        return (H_centroid,W_centroid)
    
    def compute_size(self,attention_store):
        maps=self.get_maps(attention_store).squeeze()
        H,W=maps.shape[1:]
        size=torch.sum(maps,dim=(1, 2))
        return size

    def compute_appearance(self,attention_store,activations):
        maps=self.get_maps(attention_store).squeeze()
        activations=self.get_activations()
        appearance=None
        
        pass
    
    def compute_loss(self,first_as_target=True,centroid=None,size=None,shape=None,appearance=None):
        assert first_as_target or centroid or size or shape or appearance ,"first_as_target为False时后面的属性不能全为None"
        #if first_as_target:
            
        