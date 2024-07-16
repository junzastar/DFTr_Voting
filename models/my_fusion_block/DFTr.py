import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from torch.nn import init, Sequential

class CrossAttention_csvit(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, kv):
        _, _, N_query = query.shape 
        _, _, N_kv = kv.shape 
        
        query = torch.max(query, dim=2, keepdim=True)[0].contiguous() #[B, C, 1]
        query = query.permute(0,2,1)
        kv = kv.permute(0,2,1)#[b, N, C] 
        
        B, N, C = kv.shape 

        # B, N, C = x.shape
        q = self.wq(query).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x).repeat(1,N_query,1) 
        # print("x shape: {0}".format(x.shape))
        x = (x + query).permute(0,2,1) # b,c,n
        
        return x

class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class TransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class DFTr(nn.Module):
    '''
    Deep Fusion Transformer (DFTr)
    '''

    def __init__(self, d_model, heads=2, block_exp=2,
                 n_layer=1, rgb_anchors=12800, point_anchors=3200,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.rgb_anchors = rgb_anchors
        self.point_anchors = point_anchors
        self.num_heads = heads
        d_k = d_model
        d_v = d_model
        self.pre_p2r_attn = CrossAttention_csvit(self.n_embd, num_heads=self.num_heads) #ablation study 1:Effect of bidirectional cross-modality attention
        self.pre_r2p_attn = CrossAttention_csvit(self.n_embd, num_heads=self.num_heads)
        # positional embedding parameter (learnable), rgb_feat + pts_feat
        self.pos_emb = nn.Parameter(torch.zeros(1, rgb_anchors + point_anchors, self.n_embd))
        # transformer
        self.trans_blocks = nn.Sequential(*[TransformerBlock(d_model, d_k, d_v, self.num_heads, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        # regularization
        self.drop = nn.Dropout(embd_pdrop)
        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
    
        rgb_feat = x[0]  # dim:(B, C, H*W)
        point_feat = x[1]   # dim:(B, C, n_pts)
        # print("rgb_feat, point_feat:",rgb_feat.shape, point_feat.shape)
        bs, c, hw = rgb_feat.shape
        _, _, n_pts = point_feat.shape

        rgb_feat = self.pre_p2r_attn(rgb_feat, point_feat)
        point_feat = self.pre_r2p_attn(point_feat, rgb_feat)

        rgb_feat_flat = rgb_feat.view(bs, c, -1)
        point_feat_flat = point_feat.view(bs, c, -1) 
        token_embeddings = torch.cat([rgb_feat_flat, point_feat_flat], dim=2)  # concat dim:(B, C, H*W + n_pts)
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, H*W + n_pts, C)

        x = self.drop(self.pos_emb + token_embeddings) # dim:(B, H*W + n_pts, C)
        # x = self.drop(token_embeddings)  # ablation study CMA+PE
        x = self.trans_blocks(x)  # dim:(B, H*W + n_pts, C)

        x = self.ln_f(x)  # dim:(B, H*W + n_pts, C)
        x = x.permute(0,2,1) # dim:(B, C, H*W + n_pts)

        rgb_feat_out = x[:, :, :hw].contiguous().view(bs, self.n_embd, hw)
        point_feat_out = x[:, :, hw:].contiguous().view(bs, self.n_embd, n_pts)

        return rgb_feat_out, point_feat_out # [b,c,n]

def main():
    # from common import ConfigRandLA
    # rndla_cfg = ConfigRandLA

    # n_cls = 21
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = DFTr(d_model=64, n_layer=2, rgb_anchors=128, point_anchors=256).cuda()
    rgb = torch.rand(2, 64, 128).cuda()
    point = torch.rand(2,64, 256).cuda()
    inputs = (rgb,point)
    print(model)

    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )

    rgb_feat_out, point_feat_out = model(inputs)
    print("rgb_feat_out, point_feat_out: ",rgb_feat_out.shape, point_feat_out.shape)


if __name__ == "__main__":
    main()