
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from torch.distributions import Categorical
import torch.nn.functional as F
from timm.layers.mlp import SwiGLU, Mlp 

@dataclass
class LLaMAHFConfig:
    block_size: int = 78
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    T5_xxl_dim: int = 768

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "Normal_size": dict(n_layer=12, n_head=12, n_embd=768)
}


class LLaMAHF(nn.Module):
    def __init__(self, config: LLaMAHFConfig, num_diffusion_head_layers=9, input_token_dim=16, device=torch.device('cuda'), width=1792) -> None:
        super().__init__()
        assert config.block_size is not None
        self.config = config

        cond_dim = config.T5_xxl_dim
         
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_token_dim, config.n_embd),   
                cond_embed=nn.Linear(cond_dim, config.n_embd),  
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
                )
            )
        
        target_channels = input_token_dim
        from models.diffloss import DiffLoss
        self.diff_loss = DiffLoss(
                target_channels=target_channels,
                z_channels=config.n_embd,
                width=width,
                depth=num_diffusion_head_layers,
                num_sampling_steps='50',
                grad_checkpointing=False,
            )
        self.diff_loss = self.diff_loss.to(device)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
            
    

    def forward_sample(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:
        
        text_length = clip_feature.shape[1]
        if len(idx) == 0:
            x = self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :]
        else:
            _, t = idx.size()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # forward the LLaMA model itself
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
            x = torch.cat((self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :],x), dim=1)

        for block in self.transformer.h:
            x = block(x, y_mask)
        x = self.transformer.ln_f(x)
        logits = x
        return logits

    

    def sample_for_eval_CFG(self, text, length=196, tokenize_model=None, device=torch.device('cuda'), unit_length=4, cfg=4.0):
        max_token_len = length // unit_length
        for k in range(max_token_len): 
            if k == 0:
                x = []
            else:
                x = xs
            
            feat_text = torch.from_numpy(tokenize_model.encode(text)).float()
            feat_text = feat_text.to(device)
            conditions = self.forward(x, feat_text)
            conditions = conditions[:, -1, :]                            
            
            empty_text = ''
            empty_feat_text = torch.from_numpy(tokenize_model.encode(empty_text)).float()   
            empty_feat_text = empty_feat_text.unsqueeze(0)
            empty_feat_text = empty_feat_text.to(device)
            empty_conditions = self.forward(x, empty_feat_text)       
            empty_conditions = empty_conditions[:, -1, :]                             
            temperature = 1.0
            
            # chunk
            if cfg != 1:
                mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
                sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)          
            else:   # no cfg
                scaled_logits = self.diff_loss.sample(conditions, temperature=temperature, cfg=1) 

            scaled_logits = scaled_logits.unsqueeze(0)     
            
            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

        return xs
    
    
    
    # 推理时调用，可以停止
    def sample_for_eval_CFG_inference(self, clip_text, if_categorial=False, length=312, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, threshold=3, cfg=4.5, temperature=1.0):

        import clip
        max_token_len = length // unit_length
        #print(f'Max_token_len: {max_token_len}')

        if tokenizer == 'clip':
            text = clip.tokenize(clip_text, truncate=True).to(device)  # len(cliptext)=32, text.shape=torch.Size([32, 77])

            feat_clip_text = clip_model.encode_text(text).float()  # feat_clip_text.shape=torch.Size([32, 512])
        elif tokenizer == 't5-xxl':
            feat_clip_text = torch.from_numpy(clip_model.encode(clip_text)).float()   # torch.Size([32, 768])
            #feat_clip_text = feat_clip_text.unsqueeze(0)
            feat_clip_text = feat_clip_text.to(device)

        empty_clip_text = ''
        if tokenizer == 'clip':
            empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
            empty_feat_clip_text = clip_model.encode_text(empty_text).float()
        elif tokenizer == 't5-xxl':
            empty_feat_clip_text = torch.from_numpy(clip_model.encode(empty_clip_text)).float()   # torch.Size([32, 768])
            empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
            empty_feat_clip_text = empty_feat_clip_text.to(device)
        
        for k in range(max_token_len): 
            if k == 0:
                x = []
            else:
                x = xs
            
            try:
                conditions = self.forward(x, feat_clip_text)
            except:
                conditions = self.forward(x, feat_clip_text.unsqueeze(0))

            
            conditions = conditions[:, -1, :]                            

            empty_conditions = self.forward(x, empty_feat_clip_text)       
            empty_conditions = empty_conditions[:, -1, :]                             
            
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)           
            else:
                scaled_logits = sampled_token_latent

            scaled_logits = scaled_logits.unsqueeze(0)      
            
            if reference_end_token is not None:
                distance_l2 = torch.sqrt(torch.sum((scaled_logits - reference_end_token)**2))
                print(distance_l2)
                if distance_l2 < threshold:
                    break

            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

        return xs
    def sample_for_eval_CFG_inference2(self, feat_clip_text, empty_feat_clip_text, if_categorial=False, length=312, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, threshold=3, cfg=4.5, temperature=1.0):
       
        import clip
        max_token_len = length // unit_length
        
        for k in range(max_token_len): 
            if k == 0:
                x = []
            else:
                x = xs

            try:
                conditions = self.forward(x, feat_clip_text)
            except:
                conditions = self.forward(x, feat_clip_text.unsqueeze(0))
            
            
            conditions = conditions[:, -1, :]                            
            
            

            empty_conditions = self.forward(x, empty_feat_clip_text)       
            empty_conditions = empty_conditions[:, -1, :]                             
            
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)           
            else:
                scaled_logits = sampled_token_latent

            scaled_logits = scaled_logits.unsqueeze(0)      
            
            if reference_end_token is not None:
                distance_l2 = torch.sqrt(torch.sum((scaled_logits - reference_end_token)**2))
                print(distance_l2)
                if distance_l2 < threshold:
                    break

            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

        return xs

    def sample_for_eval_CFG_inference_next_one(self, current_token=[], feat_clip_text=None, empty_feat_clip_text=None, if_categorial=False, length=312, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, threshold=3, cfg=4.5, temperature=1.0):
        
        import clip
        max_token_len = length // unit_length
       
        
        for k in range(1): 
            
            if current_token == []:
                x = []
            else:
                x = torch.cat(current_token, dim=1)
            
            
            try:
                conditions = self.forward(x, feat_clip_text)
            except:
                conditions = self.forward(x, feat_clip_text.unsqueeze(0))
            
            
            conditions = conditions[:, -1, :]                             
            

            empty_conditions = self.forward(x, empty_feat_clip_text)      
            empty_conditions = empty_conditions[:, -1, :]                         

            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)   
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)         
            else:
                scaled_logits = sampled_token_latent


            scaled_logits = scaled_logits.unsqueeze(0)    
            

            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

        return xs
    

    def sample_for_eval_CFG_babel(self, A_text, B_text, A_motion, if_categorial=False, length=6400, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, cfg=7.0, threshold=3):

        import clip
        B_token_length = length // unit_length - A_motion.shape[0]

        if tokenizer == 'clip':
            A_text = clip.tokenize(A_text, truncate=True).to(device) 
            A_feat_clip_text = clip_model.encode_text(A_text).float()  
            B_text = clip.tokenize(B_text, truncate=True).to(device)
            B_feat_clip_text = clip_model.encode_text(B_text).float()
        elif tokenizer == 't5-xxl':
            A_feat_clip_text = torch.from_numpy(clip_model.encode(A_text)).float()
            A_feat_clip_text = A_feat_clip_text.to(device)
            B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
            B_feat_clip_text = B_feat_clip_text.to(device)
        
        A_text_embeddings = self.transformer.cond_embed(A_feat_clip_text).unsqueeze(0)
        B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)

        A_motion = A_motion.unsqueeze(0)
        A_motion_embeddings = self.transformer.wte(A_motion)
        B_motion = torch.tensor([]).to(device)

        for k in range(B_token_length): 
            if k == 0:
                x = torch.cat([A_text_embeddings, A_motion_embeddings, B_text_embeddings], dim=1)
            else:
                x = xs

            
            conditions = self.forward_babel_eval(x)
            conditions = conditions[:, -1, :]                       

            empty_clip_text = ''
            if tokenizer == 'clip':
                empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
                empty_feat_clip_text = clip_model.encode_text(empty_text).float()
            elif tokenizer == 't5-xxl':
                empty_feat_clip_text = torch.from_numpy(clip_model.encode(empty_clip_text)).float()   
                empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
                empty_feat_clip_text = empty_feat_clip_text.to(device)

            empty_feat_clip_text_embedding = self.transformer.cond_embed(empty_feat_clip_text).unsqueeze(0)
            
            if k == 0:
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, empty_feat_clip_text_embedding], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)
            else:
                B_motion_embeddings = self.transformer.wte(B_motion)
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, empty_feat_clip_text_embedding, B_motion_embeddings], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)      
            
            empty_conditions = empty_conditions[:, -1, :]                            
            temperature = 1.0
            
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)    
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)           
            else:
                scaled_logits = sampled_token_latent


            scaled_logits = scaled_logits.unsqueeze(0)      
            

            B_motion = torch.cat((B_motion, scaled_logits), dim=1)

            scaled_logits_embedding = self.transformer.wte(scaled_logits)
            xs = torch.cat((x, scaled_logits_embedding), dim=1)
            

        return xs, B_motion

    def sample_for_eval_CFG_babel_inference(self, A_text, B_text, A_motion, if_categorial=False, length=6400, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, cfg=7.0, threshold=3):

        import clip
        B_token_length = length // unit_length - A_motion.shape[0]

        if tokenizer == 'clip':
            A_text = clip.tokenize(A_text, truncate=True).to(device)  
            A_feat_clip_text = clip_model.encode_text(A_text).float()  
            B_text = clip.tokenize(B_text, truncate=True).to(device)
            B_feat_clip_text = clip_model.encode_text(B_text).float()
        elif tokenizer == 't5-xxl':
            A_feat_clip_text = torch.from_numpy(clip_model.encode(A_text)).float()   
            A_feat_clip_text = A_feat_clip_text.to(device)
            B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
            B_feat_clip_text = B_feat_clip_text.to(device)
        
        A_text_embeddings = self.transformer.cond_embed(A_feat_clip_text).unsqueeze(0)
        A_text_embeddings = A_text_embeddings.unsqueeze(0)
        B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)
        B_text_embeddings = B_text_embeddings.unsqueeze(0)

        A_motion = A_motion.unsqueeze(0)
        A_motion_embeddings = self.transformer.wte(A_motion)
        B_motion = torch.tensor([]).to(device)

        attention_weights = []

        for k in range(B_token_length): 
            if k == 0:    
                x = torch.cat([A_text_embeddings, A_motion_embeddings, B_text_embeddings], dim=1)
                
            else:
                x = xs

            
            
            conditions = self.forward_babel_eval(x, return_attention=False)
            conditions = conditions[:, -1, :]                            

            empty_clip_text = ''
            if tokenizer == 'clip':
                empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
                empty_feat_clip_text = clip_model.encode_text(empty_text).float()
            elif tokenizer == 't5-xxl':
                empty_feat_clip_text = torch.from_numpy(clip_model.encode(empty_clip_text)).float()   
                empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
                empty_feat_clip_text = empty_feat_clip_text.to(device)

            empty_feat_clip_text_embedding = self.transformer.cond_embed(empty_feat_clip_text).unsqueeze(0)
            
            if k == 0:    
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, empty_feat_clip_text_embedding], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)
            else:
                B_motion_embeddings = self.transformer.wte(B_motion)
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, empty_feat_clip_text_embedding, B_motion_embeddings], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)

            empty_conditions = empty_conditions[:, -1, :]                             
            temperature = 1.0
            
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)    
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg) 

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)        
            else:
                scaled_logits = sampled_token_latent

            scaled_logits = scaled_logits.unsqueeze(0)     
            
            if reference_end_token is not None:
                distance_l2 = torch.sqrt(torch.sum((scaled_logits - reference_end_token)**2))
                print(distance_l2)
                if distance_l2 < threshold:
                    break
            
            B_motion = torch.cat((B_motion, scaled_logits), dim=1)

            scaled_logits_embedding = self.transformer.wte(scaled_logits)
            xs = torch.cat((x, scaled_logits_embedding), dim=1)
            
        
        
        return xs, B_motion
    

    def sample_for_eval_CFG_babel_inference_new(self, B_text, A_motion, if_categorial=False, length=78, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, cfg=4.5, threshold=3):

        import clip 
        B_token_length = length // unit_length

        if tokenizer == 'clip':
            A_text = clip.tokenize(A_text, truncate=True).to(device)  
            A_feat_clip_text = clip_model.encode_text(A_text).float()  
            B_text = clip.tokenize(B_text, truncate=True).to(device)
            B_feat_clip_text = clip_model.encode_text(B_text).float()
        elif tokenizer == 't5-xxl':
            B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
            B_feat_clip_text = B_feat_clip_text.to(device)

        empty_clip_text = ''
        if tokenizer == 'clip':
            empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
            empty_feat_clip_text = clip_model.encode_text(empty_text).float()
        elif tokenizer == 't5-xxl':
            empty_feat_clip_text = torch.from_numpy(clip_model.encode(empty_clip_text)).float()   
            empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
            empty_feat_clip_text = empty_feat_clip_text.to(device)
        
        B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)

        A_motion = A_motion.unsqueeze(0)
        A_motion_embeddings = self.transformer.wte(A_motion)
        B_motion = torch.tensor([]).to(device)

    
        attention_weights = []

        for k in range(B_token_length): 
            if k == 0: 
                x = torch.cat([B_text_embeddings, A_motion_embeddings], dim=1)
            else:
                x = xs

            conditions = self.forward_babel_eval(x, return_attention=False)
            conditions = conditions[:, -1, :]                             
            

            empty_feat_clip_text_embedding = self.transformer.cond_embed(empty_feat_clip_text).unsqueeze(0)
            
            if k == 0:
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings], dim=1)
                
                empty_conditions = self.forward_babel_eval(empty_input)
            else:
                B_motion_embeddings = self.transformer.wte(B_motion)
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, B_motion_embeddings], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)

            empty_conditions = empty_conditions[:, -1, :]                             
            temperature = 1.0
           
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg) 

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)       
            else:
                scaled_logits = sampled_token_latent

            scaled_logits = scaled_logits.unsqueeze(0)      
            
            if reference_end_token is not None:
                distance_l2 = torch.sqrt(torch.sum((scaled_logits - reference_end_token)**2))
                print(distance_l2)
                if distance_l2 < threshold:
                    break
            
            B_motion = torch.cat((B_motion, scaled_logits), dim=1)

            scaled_logits_embedding = self.transformer.wte(scaled_logits)
            xs = torch.cat((x, scaled_logits_embedding), dim=1)
            
        
        
        return xs, B_motion


    def sample_for_eval_CFG_babel_inference_new_demo(self, B_text, A_motion, if_categorial=False, length=312, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4, reference_end_token=None, cfg=4.5, threshold=3, temperature=1.0):

        import clip 
        B_token_length = length // unit_length - A_motion.shape[0]
        
        if tokenizer == 'clip':
            A_text = clip.tokenize(A_text, truncate=True).to(device) 
            A_feat_clip_text = clip_model.encode_text(A_text).float()  
            B_text = clip.tokenize(B_text, truncate=True).to(device)
            B_feat_clip_text = clip_model.encode_text(B_text).float()
        elif tokenizer == 't5-xxl':
            B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
            B_feat_clip_text = B_feat_clip_text.to(device)

        empty_clip_text = ''
        if tokenizer == 'clip':
            empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
            empty_feat_clip_text = clip_model.encode_text(empty_text).float()
        elif tokenizer == 't5-xxl':
            empty_feat_clip_text = torch.from_numpy(clip_model.encode(empty_clip_text)).float()   
            empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
            empty_feat_clip_text = empty_feat_clip_text.to(device)
        
        B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)
        B_text_embeddings = B_text_embeddings.unsqueeze(0)

        A_motion = A_motion.unsqueeze(0)
        A_motion_embeddings = self.transformer.wte(A_motion)
        B_motion = torch.tensor([]).to(device)

        # 存储所有层的注意力权重
        attention_weights = []

        for k in range(B_token_length): 
            if k == 0: 
                x = torch.cat([B_text_embeddings, A_motion_embeddings], dim=1)
               
            else:
                x = xs

            
            conditions = self.forward_babel_eval(x, return_attention=False)
            conditions = conditions[:, -1, :]                             
            

            empty_feat_clip_text_embedding = self.transformer.cond_embed(empty_feat_clip_text).unsqueeze(0)
            
            if k == 0:
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)
            else:
                B_motion_embeddings = self.transformer.wte(B_motion)
                empty_input = torch.cat([empty_feat_clip_text_embedding, A_motion_embeddings, B_motion_embeddings], dim=1)
                empty_conditions = self.forward_babel_eval(empty_input)

            empty_conditions = empty_conditions[:, -1, :]                             
           
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)           
            else:
                scaled_logits = sampled_token_latent

            scaled_logits = scaled_logits.unsqueeze(0)      
            
            if reference_end_token is not None:
                distance_l2 = torch.sqrt(torch.sum((scaled_logits - reference_end_token)**2))
                print(distance_l2)
                if distance_l2 < threshold and k > 10:
                    break
            
            B_motion = torch.cat((B_motion, scaled_logits), dim=1)

            scaled_logits_embedding = self.transformer.wte(scaled_logits)
            xs = torch.cat((x, scaled_logits_embedding), dim=1)
            
        
        
        return xs, B_motion


    
    #--------------Test classification head--------------------
    def sample_for_eval_classification(self, clip_text, if_categorial=False, length=196, clip_model=None, device=torch.device('cuda'), tokenizer='clip', unit_length=4):
        
        import clip
       
        
        for k in range(51): 
            if k == 0:
                x = []
            else:
                x = xs

            if tokenizer == 'clip':
                text = clip.tokenize(clip_text, truncate=True).to(device)  

                feat_clip_text = clip_model.encode_text(text).float() 
            elif tokenizer == 't5-xxl':
                feat_clip_text = torch.from_numpy(clip_model.module.encode(clip_text)).float()  
           
            conditions = self.forward(x, feat_clip_text)
            conditions = conditions[:, -1, :]     

            empty_clip_text = ''
            if tokenizer == 'clip':
                empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
                empty_feat_clip_text = clip_model.encode_text(empty_text).float()
            elif tokenizer == 't5-xxl':
                empty_feat_clip_text = torch.from_numpy(clip_model.module.encode(empty_clip_text)).float()  
                empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
                empty_feat_clip_text = empty_feat_clip_text.to(device)

            empty_conditions = self.forward(x, empty_feat_clip_text)     
            empty_conditions = empty_conditions[:, -1, :]                    
           
            temperature = 1.0
            cfg = 7.5
            
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0) 
            sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

            # chunk
            if cfg != 1:
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)        
            else:
                scaled_logits = sampled_token_latent


            prediction_logits = self.classify_head(conditions)  
            probs = torch.sigmoid(prediction_logits)
            predicted_classes = torch.argmax(probs, dim=-1)
            

            scaled_logits = scaled_logits.unsqueeze(0)      
                
            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

            if predicted_classes == 1:
                break

        return xs
    

    #--------------------Test CFG-----------------------
    def sample_for_eval_CFG_test(self, clip_text, if_categorial=False, length=196, clip_model=None, cfg=1, device=torch.device('cuda'), tokenizer='clip', unit_length=4):

        import clip
        max_token_len = length // unit_length
        
        
        for k in range(max_token_len): 
            if k == 0:
                x = []
            else:
                x = xs

            
            if cfg != 1:
                if tokenizer == 'clip':
                    text = clip.tokenize(clip_text, truncate=True).to(device)  

                    feat_clip_text = clip_model.encode_text(text).float() 
                elif tokenizer == 't5-xxl':
                    feat_clip_text = torch.from_numpy(clip_model.module.encode(clip_text)).float()  
                
                conditions = self.forward(x, feat_clip_text)       
                
                conditions = conditions[:, -1, :]                            
                empty_clip_text = ''
                if tokenizer == 'clip':
                    empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
                    empty_feat_clip_text = clip_model.encode_text(empty_text).float()
                elif tokenizer == 't5-xxl':
                    empty_feat_clip_text = torch.from_numpy(clip_model.module.encode(empty_clip_text)).float()   
                    empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
                    empty_feat_clip_text = empty_feat_clip_text.to(device)

                empty_conditions = self.forward(x, empty_feat_clip_text)     
                empty_conditions = empty_conditions[:, -1, :]                             
                temperature = 1.0
                
                
                mix_conditions = torch.cat([conditions, empty_conditions], dim=0)     
                sampled_token_latent = self.diff_loss.sample(mix_conditions, temperature=temperature, cfg=cfg)  

                # chunk
                scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)           
            
            else:   
                if tokenizer == 'clip':
                    text = clip.tokenize(clip_text, truncate=True).to(device)  
                    feat_clip_text = clip_model.encode_text(text).float() 
                elif tokenizer == 't5-xxl':
                    feat_clip_text = torch.from_numpy(clip_model.module.encode(clip_text)).float()  
                    feat_clip_text = feat_clip_text.to(device)


                conditions = self.forward(x, feat_clip_text)       
  
                conditions = conditions[:, -1, :]                             
                temperature = 1.0
                sampled_token_latent = self.diff_loss.sample(conditions, temperature=temperature, cfg=cfg) 
                scaled_logits = sampled_token_latent       

            scaled_logits = scaled_logits.unsqueeze(0)      
                
            if k == 0:
                xs = scaled_logits
            else:
                xs = torch.cat((xs, scaled_logits), dim=1)

        return xs
    #--------------------------------------------------

    def forward_discrete(self, idx: torch.Tensor, clip_feature: torch.Tensor, use_cache=False, past_key_values=None) -> torch.Tensor:
        if len(idx) == 0:
            token_embeddings = self.transformer.cond_embed(clip_feature).unsqueeze(0)

        else:
            b, t = idx.size()
            #idx = idx.float()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the LLaMA model itself
            token_embeddings = self.transformer.wte(idx)  
            text_embeddings = self.transformer.cond_embed(clip_feature).unsqueeze(1)        
            token_embeddings = torch.cat([text_embeddings, token_embeddings], dim=1) 
        
        x = token_embeddings

        # -------------------kv cache-------------------
        #presents = () if use_cache else None
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.transformer.h)   
            

        for i,block in enumerate(self.transformer.h):
            if use_cache:
                last_past = past_key_values[i]     
                x, presents = block(x, last_past, use_cache)
                past_key_values[i] = list(presents)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) 
        

        return logits


    def forward(self, idx: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if len(idx) == 0:
            token_embeddings = self.transformer.cond_embed(feature).unsqueeze(0)

        else:
            b, t, c = idx.size()
            idx = idx.float()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the LLaMA model itself
            token_embeddings = self.transformer.wte(idx)  
            text_embeddings = self.transformer.cond_embed(feature).unsqueeze(1)    
            token_embeddings = torch.cat([text_embeddings, token_embeddings], dim=1) 
        
        x = token_embeddings  

        for i,block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)    
        logits = self.out_proj(x)
        return logits
        

    def babel_long(self, idx: torch.Tensor, clip_feature: torch.Tensor, use_cache=False, past_key_values=None, num_subseq=None, length=None) -> torch.Tensor:
        
        b, t, c = idx.size()
        idx = idx.float()
        idx = self.transformer.wte(idx)
        assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        for i in range(b):
            length_i = length[i][:num_subseq[i]]       
            clip_feature_i = clip_feature[i][:num_subseq[i]]

            pointer = 0
            for j in range(num_subseq[i]):
                if j > 0:
                    pointer += length_i[j].item()
                    pointer += 1
                pointer = int(pointer)

                clip_feature_i_j = self.transformer.cond_embed(clip_feature_i[j].unsqueeze(0)).unsqueeze(1)
                idx[i] = torch.cat([idx[i][:pointer].unsqueeze(0), clip_feature_i_j, idx[i][pointer:-1].unsqueeze(0)], dim=1)[0]
                
        x = idx


        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.transformer.h)   
            

        for i,block in enumerate(self.transformer.h):
            if use_cache:
                last_past = past_key_values[i]      
                x, presents = block(x, last_past, use_cache)
                past_key_values[i] = list(presents)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.out_proj(x)
        return logits
        

    def forward_babel_eval(self, x, return_attention=False) -> torch.Tensor:
        layer_attentions = []
        for block in self.transformer.h:
            if return_attention:
                x, att = block(x, return_attention=True)
                layer_attentions.append(att)
            else:
                x = block(x)
        
        x = self.transformer.ln_f(x)
        if self.use_out_proj:
            logits = self.out_proj(x)
        else:
            logits = x
        
        if return_attention:
            return logits, layer_attentions
        return logits
    
    def forward_babel(self, idx: torch.Tensor, clip_feature: torch.Tensor, A_token_length) -> torch.Tensor:
        if len(idx) == 0:   # inference
            token_embeddings = self.transformer.cond_embed(clip_feature).unsqueeze(1)

        else:
            b, t, c = idx.size()
            idx = idx.float()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            

            A_feature = clip_feature[:, 0, :]             
            B_feature = clip_feature[:, 1, :]                   


            A_text_embeddings = self.transformer.cond_embed(A_feature).unsqueeze(1)
            B_text_embeddings = self.transformer.cond_embed(B_feature).unsqueeze(1)

            token_embeddings = torch.zeros(b, self.config.block_size, self.config.n_embd).to(idx.device)
            for i in range(b):
                A_idx = idx[i, :A_token_length[i].item(), :]
                B_idx = idx[i, A_token_length[i].item():-2, :]
                token_embeddings[i, :, :] = torch.cat([A_text_embeddings[i], self.BOM_tag, self.transformer.wte(A_idx), B_text_embeddings[i], self.BOM_tag, self.transformer.wte(B_idx)], dim=0)  #token_embeddings.shape = (b,t+1,1024)
        
        x = token_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.use_out_proj:
            logits = self.out_proj(x)
        else:
            logits = x


        return logits

    def forward_babel2(self, idx: torch.Tensor, clip_feature: torch.Tensor) -> torch.Tensor:
        if len(idx) == 0:   # inference
            token_embeddings = self.transformer.cond_embed(clip_feature).unsqueeze(1)

        else:
            b, t, c = idx.size()
            idx = idx.float()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            B_feature = clip_feature
            B_text_embeddings = self.transformer.cond_embed(B_feature)

            idx_embeddings = self.transformer.wte(idx)

            
            token_embeddings = torch.cat([B_text_embeddings, idx_embeddings], dim=1)
            
        
        x = token_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.use_out_proj:
            logits = self.out_proj(x)
        else:
            logits = x

        return logits
    

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, using_old_initilization: bool = False
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]

        # Tie weights again if needed
        # self.tie_weights()

        return model_embeds
    
    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            # if is_deepspeed_zero3_enabled():
            #     import deepspeed

            #     with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
            #         new_num_tokens = new_embeddings.weight.shape[0]
            # else:
            new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        # if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        if self.get_output_embeddings() is not None and not False:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # if hasattr(old_lm_head, "_hf_hook"):
            #     hook = old_lm_head._hf_hook
            #     add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            print(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
        if old_num_tokens == new_num_tokens and not False:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings


    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
        if old_num_tokens == new_num_tokens and not False:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAHFConfig.from_name(name))


class Block(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)

        # sentence level:
        self.attn = CausalSelfAttention(config)        
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, last_past=None, use_cache=False, return_attention=False) -> torch.Tensor:
        if use_cache:  
            if return_attention:
                a, attn = self.attn.forward_attn(self.rms_1(x), last_past, use_cache)
            else:
                a, present = self.attn(self.rms_1(x), last_past, use_cache)
            x = x + a
        else:
            if return_attention:
                a, attn = self.attn.forward_attn(self.rms_1(x))
            else:
                a = self.attn(self.rms_1(x))
            x = x + a
        x = x + self.mlp(self.rms_2(x))

        if use_cache:
            if return_attention:
                return x, present, attn
            else:
                return x, present  
        else:
            if return_attention:
                return x, attn
            else:
                return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

        def scaling_factor(sequence_threshold):
            return np.log2((sequence_threshold**2) - sequence_threshold)
        scale_init = scaling_factor(self.block_size) 
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, x: torch.Tensor, last_past=None, use_cache=False) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # kv_cache
        if use_cache:
            if last_past is not None:
                past_key, past_value = last_past
                k = torch.cat([past_key, k], dim=-2)  
                v = torch.cat([past_value, v], dim=-2)   
            # else:
            #     key_states = k
            #     value_states = v

        if use_cache:
            present = (k, v)
        else:
            present = None

        # QK-Norm
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )


        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)


        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scale.item())
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        
        if use_cache:
            return y, present
        return y

    def forward_attn(self, x: torch.Tensor, last_past=None, use_cache=False) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # kv_cache
        if use_cache:
            if last_past is not None:
                past_key, past_value = last_past
                k = torch.cat([past_key, k], dim=-2)  
                v = torch.cat([past_value, v], dim=-2) 
            # else:
            #     key_states = k
            #     value_states = v

        if use_cache:
            present = (k, v)
        else:
            present = None

        # QK-Norm
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )


        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)  # [B, n_head, T, T]

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        
        return y, att

class LengthCausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )


        # q: 1, 16, 40 ,64
        # q: 128, 16, 106, 64
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)
 
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        attn_mask = torch.tril(attn_mask) 
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)

        text_mask = y_mask.unsqueeze(2)*y_mask.unsqueeze(1)
        text_mask = F.pad(text_mask, (0, T-y_mask.shape[1], 0, T-y_mask.shape[1]), mode='constant', value=0)
        attn_mask = torch.logical_or(attn_mask, text_mask)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1), dropout_p=0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C) 

     
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """用了SwiGLU"""
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = (
        torch.float32 if dtype in dtypes_requiring_casting else dtype
    )
    complex_dtype = (
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]
    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
