from builtins import print
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet import PSPNet
import models.pytorch_utils as pt_utils
from models.RandLA.RandLANet import Network as RandLANet
# from models.pointnet2_utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
# from models.restormer.restormer import R_TransformerBlock
from models.my_fusion_block.DFTr import DFTr

from einops import rearrange
import numpy as np


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}

class SA_Layer(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(SA_Layer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class SA_Layer_pct(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_pct, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): #[b, c, n]
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channel, num_heads = 4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.dim = in_channel
        self.out_dim = in_channel
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5

        self.wq_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wk_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wv_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        # self.wq_point = nn.Linear(in_channel, in_channel, bias=qkv_bias)
        # self.wk_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        # self.wv_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_r2p = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        # self.proj_p2r = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        

        # self.trans_conv = nn.Conv1d(channel, channel, 1)
        # self.norm = nn.BatchNorm1d(in_channel)
        # self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_emb, geo_emb): #[b, c, n] 
        rgb_emb = rgb_emb.permute(0,2,1)
        geo_emb = geo_emb.permute(0,2,1)#[b, N, C] 
        _, N_rgb, _ = rgb_emb.shape #2048
        B, N_geo, C = geo_emb.shape #2048
        # print("rgb, geo shape:{0},{1}".format(rgb_emb.shape,geo_emb.shape))
        #from point to rgb
        q_rgb = self.wq_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        k_point = self.wk_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,8,2048,32
        v_point = self.wv_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        # print("q k v shape:{0},{1},{2}".format(q.shape,k.shape,v.shape))
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # scale_p2r = np.sqrt(k_point.size(1))
        scale_p2r = self.scale
        
        attn_p2r = (q_rgb.transpose(-2, -1) @ k_point) * scale_p2r # B,8,32,32
        
        attn_p2r = self.softmax(attn_p2r) # B,4,32,32
        attn_p2r = self.attn_drop(attn_p2r)
        # print("attn shape: {0}".format(attn.shape))

        res_emb_p2r = (v_point @ attn_p2r).transpose(1,2).reshape(B, N_rgb, C) # B,2048,32*8
        
        res_emb_p2r = self.proj_drop(self.proj_p2r(res_emb_p2r)) # B, c, N_rgb
        # x_rs = self.proj_drop(x_rs).permute(0, 2, 1).contiguous()# B, C,N
        # Att_emb = Att_emb + x_rs # B, C, N
        # x_rs = x_rs + Att_emb
        #from rgb to point
        q_point = self.wq_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        k_rgb = self.wk_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,C,N
        v_rgb = self.wv_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        scale_r2p = self.scale

        attn_r2p = (q_point.transpose(-2, -1) @ k_rgb) * scale_r2p # B,N,N
        attn_r2p = self.softmax(attn_r2p) # B,N,N
        attn_r2p = self.attn_drop(attn_r2p)
        res_emb_r2p = (v_rgb @ attn_r2p).transpose(1,2).reshape(B, N_geo, C) # B, C, N
        
        res_emb_r2p = self.proj_drop(self.proj_r2p(res_emb_r2p)) # B, c, N_geo

        rgb_emb_att = (rgb_emb + res_emb_p2r).permute(0, 2, 1)
        geo_emb_att = (geo_emb + res_emb_r2p).permute(0, 2, 1)
        
        res = torch.cat([rgb_emb_att, geo_emb_att],dim=1) # B, 512, 2048
        return res

class C_CrossAttention(nn.Module):
    def __init__(self, in_channel, num_heads = 8, qkv_bias=False, attn_drop=0.15, proj_drop=0.15):
        super().__init__()

        self.num_heads = num_heads
        self.dim = in_channel
        self.out_dim = in_channel
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5

        self.wq_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wk_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wv_rgb = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wq_point = nn.Linear(in_channel, in_channel, bias=qkv_bias)
        self.wk_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.wv_point = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_r2p = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.proj_p2r = nn.Linear(in_channel, in_channel,  bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        

        # self.trans_conv = nn.Conv1d(channel, channel, 1)
        # self.norm = nn.BatchNorm1d(in_channel)
        # self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_emb, geo_emb): #[b, c, n] 
        rgb_emb = rgb_emb.permute(0,2,1)
        geo_emb = geo_emb.permute(0,2,1)#[b, N, C] 
        _, N_rgb, _ = rgb_emb.shape #2048
        B, N_geo, C = geo_emb.shape #2048
        # print("rgb, geo shape:{0},{1}".format(rgb_emb.shape,geo_emb.shape))
        #from point to rgb
        q_rgb = self.wq_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        k_point = self.wk_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,8,2048,32
        v_point = self.wv_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,8,2048,32
        # print("q k v shape:{0},{1},{2}".format(q.shape,k.shape,v.shape))
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # scale_p2r = np.sqrt(k_point.size(1))
        scale_p2r = self.scale
        
        attn_p2r = (q_rgb.transpose(-2, -1) @ k_point) * scale_p2r # B,8,32,32
        
        attn_p2r = self.softmax(attn_p2r) # B,4,32,32
        attn_p2r = self.attn_drop(attn_p2r)
        # print("attn shape: {0}".format(attn.shape))

        res_emb_p2r = (v_point @ attn_p2r).transpose(1,2).reshape(B, N_rgb, C) # B,2048,32*8
        
        # res_emb_p2r = self.proj_drop(self.norm(self.proj_p2r(res_emb_p2r).permute(0,2,1))).permute(0,2,1) # B, c, N_rgb
        res_emb_p2r = self.proj_drop(self.proj_p2r(res_emb_p2r)) # B, c, N_rgb
        # x_rs = self.proj_drop(x_rs).permute(0, 2, 1).contiguous()# B, C,N
        # Att_emb = Att_emb + x_rs # B, C, N
        # x_rs = x_rs + Att_emb
        #from rgb to point
        q_point = self.wq_point(geo_emb).view(B, N_geo, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        k_rgb = self.wk_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,C,N
        v_rgb = self.wv_rgb(rgb_emb).view(B, N_rgb, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,N,C
        scale_r2p = self.scale

        attn_r2p = (q_point.transpose(-2, -1) @ k_rgb) * scale_r2p # B,N,N
        attn_r2p = self.softmax(attn_r2p) # B,N,N
        attn_r2p = self.attn_drop(attn_r2p)
        res_emb_r2p = (v_rgb @ attn_r2p).transpose(1,2).reshape(B, N_geo, C) # B, C, N
        
        # res_emb_r2p = self.proj_drop(self.norm(self.proj_r2p(res_emb_r2p).permute(0,2,1))).permute(0,2,1) # B, c, N_geo
        res_emb_r2p = self.proj_drop(self.proj_r2p(res_emb_r2p))# B, c, N_geo

        rgb_emb_att = (rgb_emb + res_emb_p2r).permute(0, 2, 1)
        geo_emb_att = (geo_emb + res_emb_r2p).permute(0, 2, 1)
        
        res = torch.cat([rgb_emb_att, geo_emb_att],dim=1) # B, 512, 2048
        return res #rgb_emb_att, geo_emb_att

class netlocalD(nn.Module):
    def __init__(self, in_channel):
        super(netlocalD,self).__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x

class netlocalG(nn.Module):
    def __init__(self, in_channel, num_points, out_channel = 512):
        super(netlocalG,self).__init__()
        self.fc1 = nn.Linear(in_channel, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(192,256)
        self.fc5 = nn.Linear(512,out_channel)

        self.ap1 = nn.AvgPool1d(num_points)

        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        # [bs, C, N_pts]
        x = x.permute(0,2,1) # [bs, N_pts, C]
        x_256_1 = F.relu(self.bn_1(self.fc1(x)))
        x_128 = F.relu(self.bn_2(self.fc2(x)))
        x_64 = F.relu(self.bn_3(self.fc3(x)))
        x_192 = torch.cat((x_128, x_64),dim=1)
        x_256_4 = F.relu(self.bn_4(self.fc4(x_192)))
        x_512 = torch.cat((x_256_1, x_256_4),dim=1)
        x_1024 = F.relu(self.bn_5(self.fc5(x_512)))
        F_Gmap = self.ap1(x_1024.permute(0,2,1)) # [bs, 1024, 1]
        return F_Gmap

class CrossAttention_csvit(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        _, _, N_query = query.shape #2048
        _, _, N_kv = kv.shape #2048
        query = F.adaptive_max_pool1d(query, 1) #[B, 1, 64]
        query = query.permute(0,2,1)
        kv = kv.permute(0,2,1)#[b, N, C] 
        
        B, N, C = kv.shape #2048
        
        # print("query shape: {0}".format(query.shape))

        # B, N, C = x.shape
        q = self.wq(query).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x).repeat(1,N_query,1).permute(0,2,1) # b,c,n
        # print("x shape: {0}".format(x.shape))
        
        return x

class FFB6D(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, n_kps=8, image_anchors=480*640
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        cnn = psp_models['resnet34'.lower()]() #34

        rndla = RandLANet(rndla_cfg)

        self.cnn_pre_stages = nn.Sequential(
            #ZJ input [bs,3,192,192]
            cnn.feats.conv1,  # stride = 2, [bs, 64, 96, 96]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 48, 48]
        ) # 进入image network
        self.rndla_pre_stages = rndla.fc0 #进入Point network [N,8]
        
        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 96, 96]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 48, 48]
            # stride = 1, [bs, 256, 48, 48]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4), # stride = 1, [bs, 512, 48, 48]
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 48, 48]
        ]) #CNN downsample 结束
        self.ds_sr = [4, 8, 8, 8] #downsample 的比例 总共四个下采样层

        self.rndla_ds_stages = rndla.dilated_res_blocks

        self.ds_rgb_oc = [64, 128, 512, 1024] #Image branch 下采样层输出的通道
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out] #d_out = [32, 64, 128, 256]  # feature dimension
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        # self.ds_sa_fuse_r2p_layers = nn.ModuleList() # self attention 层
        # self.ds_sa_fuse_p2r_layers = nn.ModuleList() # self attention 层
        # self.ds_cs_fuse_r2p_layers = nn.ModuleList() # cross attention 层
        # self.ds_cs_fuse_p2r_layers = nn.ModuleList() # cross attention 层
        self.ds_dftr_layer = nn.ModuleList() # Deep fusion transfomer layer
        num_fusion_layer_ds = [1,1,1,1]
        ds_rgb_anchor = [image_anchors // 16, image_anchors // 64, image_anchors // 64, image_anchors // 64]
        ds_pre_fusion_dim = [128,256,1024,2048]

        for i in range(4):
            if i < 3:
                continue
            self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], ds_pre_fusion_dim[i], kernel_size=(1, 1),
                    bn=True
                ) #ZJ # 1. channel 64 -> 128; 2. channel 128 ->  256
                  # 3. channel 512 -> 1024; 4. channel 1024 -> 2048
            )

            self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], ds_pre_fusion_dim[i], kernel_size=(1, 1),
                    bn=True
                ) #ZJ # 1. channel 64 -> 128; 2. channel 128 -> 256
                  #  3. channel 256 -> 1024; 4. channel 512 -> 2048
            )
            ######################### add SA & cross attention layer #################
            # self.ds_sa_fuse_r2p_layers.append(
            #     # SA_Layer(self.ds_rgb_oc[i])
            #     R_TransformerBlock(self.ds_rgb_oc[i],num_heads=heads_ds[i],ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias')
            # )
            # self.ds_cs_fuse_r2p_layers.append(
            #     CrossAttention_csvit(self.ds_rndla_oc[i],num_heads=heads_ds[i])
            # )
            self.ds_dftr_layer.append(
                DFTr(d_model=ds_pre_fusion_dim[i],n_layer=num_fusion_layer_ds[i],rgb_anchors=ds_rgb_anchor[i],point_anchors=rndla_cfg.num_sub_points[i])
            )
            ######################### add SA & cross attention layer #################
            self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    ds_pre_fusion_dim[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                ) #ZJ # 1. channel 128 -> 64; 2. channel 256 -> 128
                      # 3. channel 1024 -> 512; 4. channel 2048 -> 1024
            )

            
            ######################### add SA & cross attention layer #################
            # self.ds_sa_fuse_p2r_layers.append(
            #     SA_Layer_pct(self.ds_rgb_oc[i])
            # )
            # self.ds_cs_fuse_p2r_layers.append(
            #     CrossAttention_csvit(self.ds_rgb_oc[i],num_heads=heads_ds[i])
            # )
            ######################### add SA & cross attention layer #################
            self.ds_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    ds_pre_fusion_dim[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )#ZJ # 1. channel 128 -> 64; 2. channel 256 -> 128
                  #  3. channel 1024 -> 256; 4. channel 2048 -> 512
            )
        self.ds_dftr_layer = nn.Sequential(*self.ds_dftr_layer)
        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 96, 96]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 128, 192, 192]
            # nn.Sequential(cnn.before_final),  # [bs, 64, 192, 192]
            # nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 192, 192]
            # nn.Sequential(cnn.up_3, cnn.drop_2),  # [bs, 64, 192, 192]
            # nn.Sequential(cnn.final)  # [bs, 64, 192, 192]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = [] #[256, 128, 64, 64]  # feature dimension
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2]) #d_out = [32, 64, 128, 256]  # feature dimension
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        # self.up_sa_fuse_r2p_layers = nn.ModuleList() # SA attention 层
        # self.up_sa_fuse_p2r_layers = nn.ModuleList() # SA attention 层
        # self.up_cs_fuse_r2p_layers = nn.ModuleList() # cross attention 层
        # self.up_cs_fuse_p2r_layers = nn.ModuleList() # cross attention 层
        self.up_dftr_layer = nn.ModuleList() # Deep fusion transfomer layer
        num_fusion_layer_up = [1,1,1]
        up_pre_fusion_dim = [512,128,128]
        up_rgb_anchor = [image_anchors // 16, image_anchors // 4, image_anchors // 1]

        for i in range(n_fuse_layer):
            if i > 0:
                continue
            self.up_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], up_pre_fusion_dim[i], kernel_size=(1, 1),
                    bn=True
                )
                #ZJ # 1. channel 256 -> 512; 2. channel 64 -> 128
                  #  3. channel 64 -> 128; 
            )

            self.up_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i], up_pre_fusion_dim[i], kernel_size=(1, 1),
                    bn=True
                )
                #ZJ # 1. channel 256 -> 512; 2. channel 128 -> 128
                  #  3. channel 64 -> 128; 
            )

            ######################### add SA & cross attention layer #################
            # self.up_sa_fuse_r2p_layers.append(
            #     # SA_Layer(self.up_rgb_oc[i])
            #     R_TransformerBlock(self.up_rgb_oc[i],num_heads=heads_up[i],ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias')
            # )
            # self.up_cs_fuse_r2p_layers.append(
            #     CrossAttention_csvit(self.up_rndla_oc[i],num_heads=heads_up[i])
            # )
            self.up_dftr_layer.append(
                DFTr(d_model=up_pre_fusion_dim[i], n_layer=num_fusion_layer_up[i],rgb_anchors=up_rgb_anchor[i],point_anchors=rndla_cfg.num_sub_points[-i-2])
            )
            ######################### add SA & cross attention layer #################
            self.up_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    up_pre_fusion_dim[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
                #ZJ # 1. channel 512 -> 256; 2. channel 256 -> 128
                  #  3. channel 128 -> 64; 
            )

            
            ######################### add SA & cross attention layer #################
            # self.up_sa_fuse_p2r_layers.append(
            #     SA_Layer_pct(self.up_rgb_oc[i])
            # )
            # self.up_cs_fuse_p2r_layers.append(
            #     CrossAttention_csvit(self.up_rgb_oc[i], num_heads=heads_up[i])
            # )
            ######################### add SA & cross attention layer #################
            self.up_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    up_pre_fusion_dim[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
                #ZJ # 1. channel 512 -> 256; 2. channel 128 -> 64
                  #  3. channel 128 -> 64; 
            )
        self.up_dftr_layer = nn.Sequential(*self.up_dftr_layer)
        #### FINAL Fusion layer
        # self.final_fusion_layer = C_CrossAttention(64)
        self.final_fusion_layer = DenseFusion(self.n_pts)

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgbd_seg_layer = (
            # pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            pt_utils.Seq(1664)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            # pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            pt_utils.Seq(1664)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            # pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            pt_utils.Seq(1664)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*3, activation=None)
        )

        ######################## add confidence prediction #######################
        self.ctr_ofst_score_layer = (
            # pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            pt_utils.Seq(1664)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(1, activation=None)
        )

        self.kp_ofst_score_layer = (
            # pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
            pt_utils.Seq(1664)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*1, activation=None)
        )
        ######################## add confidence prediction #######################

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features
     
    

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # stride = 1, [bs, 64, 192, 192]
        # rndla pre
        xyz, p_emb = self._break_up_pc(inputs['cld_rgb_nrm']) #xyz[bs, 3, 2048]
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb) #[bs,8, npts] zj
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1 // bs*8*2048*1

        # print("p_emb_Pre_process_stage shape: {0}".format(p_emb.shape)) #[4, 8, 12800, 1]
        # print("code run here 1！")

        # ###################### encoding stages #############################
        ds_emb = []
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size() # ids=0 [bs,64,192,192]

            # print("rgb_emb0{0} shape: {1}".format(i_ds, rgb_emb0.shape))
            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            ) #[bs,64,1024,1]

            # print("p_emb_first_encode_stage shape{0}: {1}".format(i_ds,f_encoder_i.shape))

            f_sampled_i = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds]) # [bs, 64, 512, 1]
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)
            
            # print("inputs['cld_xyz{0}] shape: {1}".format(i_ds, inputs['cld_xyz%d' % i_ds].shape))
            # print("inputs['cld_sub_idx{0}] shape: {1}".format(i_ds, inputs['cld_sub_idx%d' % i_ds].shape))
            # print("p_emb_first_encode_stage_after_samp shape{0}: {1}".format(i_ds,p_emb0.shape))
            # print("inputs['cld_xyz0'] shape: {0}".format(inputs['cld_xyz0'].shape))
            # print("inputs['cld_xyz1'] shape: {0}".format(inputs['cld_xyz1'].shape))
            # print("inputs['cld_xyz2'] shape: {0}".format(inputs['cld_xyz2'].shape))
            # print("inputs['cld_xyz3'] shape: {0}".format(inputs['cld_xyz3'].shape))
            # print("inputs['cld_xyz4'] shape: {0}".format(inputs['cld_xyz4'].shape))
            # print("inputs['cld_sub_idx0'] shape: {0}".format(inputs['cld_sub_idx0'].shape))
            # print("inputs['cld_sub_idx1'] shape: {0}".format(inputs['cld_sub_idx1'].shape))
            # print("inputs['cld_sub_idx2'] shape: {0}".format(inputs['cld_sub_idx2'].shape))
            # print("inputs['cld_sub_idx3'] shape: {0}".format(inputs['cld_sub_idx3'].shape))

            if i_ds in [3]:
                rgb_emb0_fuse = self.ds_fuse_r2p_pre_layers[i_ds-3](rgb_emb0) # [bs, 128, h,w]
                p_emb0_fuse = self.ds_fuse_p2r_pre_layers[i_ds-3](p_emb0) # [bs, 128, N, 1]
                # temp_feat = (rgb_emb0.view(bs,-1,hr*wr), p_emb0.squeeze(dim=3))
                att_fea_rgb_DS, att_fea_pts_DS = self.ds_dftr_layer[i_ds-3]((rgb_emb0_fuse.view(bs,-1,hr*wr), p_emb0_fuse.squeeze(dim=3)))
                # print("rgb_emb0 shape: {0}".format(rgb_emb0.shape))
                # print("att_fea_rgb_DS shape: {0}".format(att_fea_rgb_DS.shape))
                # print("processed shape: {0}".format(self.ds_fuse_r2p_fuse_layers[i_ds](att_fea_rgb_DS.view(bs, -1, hr, wr)).shape))
                rgb_emb = rgb_emb0 + self.ds_fuse_r2p_fuse_layers[i_ds-3](att_fea_rgb_DS.view(bs, -1, hr, wr)) #[b,c,h,w]
                p_emb = p_emb0 + self.ds_fuse_p2r_fuse_layers[i_ds-3](att_fea_pts_DS.unsqueeze(dim=3)) #[b,c,n,1]
            else:
                rgb_emb = rgb_emb0
                p_emb = p_emb0

            ds_emb.append(p_emb)
        # print("code run here 2-end encoding！")
        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages) #4
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb) # [bs,512,96,96]
            bs, c, hr, wr = rgb_emb0.size()

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )

            #print("p_emb_{0}_before UP shape: {1}".format(i_up,f_interp_i.shape))
            
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i

            #print("p_emb_{0}_After UP shape: {1}".format(i_up,p_emb0.shape))

            # fuse point feauture to rgb feature
            if i_up in [0]:
                p_emb0_fuse = self.up_fuse_p2r_pre_layers[i_up](p_emb0) #[bs,c,n,1]
                rgb_emb0_fuse = self.up_fuse_r2p_pre_layers[i_up](rgb_emb0) #[bs,c,h,w]
                # temp_feat = (rgb_emb0.view(bs,-1,hr*wr), p_emb0.squeeze(dim=3))
                att_fea_rgb_DS, att_fea_pts_DS = self.up_dftr_layer[i_up]((rgb_emb0_fuse.view(bs,-1,hr*wr), p_emb0_fuse.squeeze(dim=3)))
                rgb_emb = rgb_emb0 + self.up_fuse_r2p_fuse_layers[i_up](att_fea_rgb_DS.view(bs, -1, hr, wr)) #[b,c,h,w]
                p_emb = p_emb0 + self.up_fuse_p2r_fuse_layers[i_up](att_fea_pts_DS.unsqueeze(dim=3)) #[b,c,n,1]
            else:
                rgb_emb = rgb_emb0
                p_emb = p_emb0
            ############### Modified by ZJ R2P ##################

        # print("code run here 3-end decoding！")
        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb) #[bs,64,192,192]
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1) #[bs,64,1024,1]

        # print("rgb_emb shape: {0}.".format(rgb_emb.shape))
        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)#[bs,64,192*192]
        # print("rgb_emb_c shape: {0}.".format(rgb_emb_c.shape))
        # print("inputs['choose'] shape: {0}.".format(inputs['choose'].view(bs,1,-1).shape))
        choose_emb = inputs['choose'].view(bs,1,-1).repeat(1, di, 1) #[bs,64,1024]
        # max_num = torch.max(inputs['choose'])
        # if max_num > 36865:
        #     print("max_num!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: {0}".format(max_num))
            
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()

        # if True in np.isnan(rgb_emb_c.cpu().detach().numpy()):
        #         print("rgb_emb_c is nan*******************8: {0}.".format(rgb_emb_c))
        #         print("p_emb shape: {0}.".format(p_emb))

        # print("Final shape!!!!!!!!!!!!!!. rgb_emb_c: {0}. p_emb: {1}".format(rgb_emb_c.shape, p_emb.shape))
        # print("Final shape!!!!!!!!!!!!!!. rgb_emb_c: {0}. p_emb: {1}".format(rgb_emb_c, p_emb))
        

        # Use DenseFusion in final layer, which will hurt performance due to overfitting
        rgbd_emb = self.final_fusion_layer(rgb_emb_c, p_emb)

        # Use simple concatenation. Good enough for fully fused RGBD feature.
        # rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1) #[bs,128,1024] #-<<<<<<<<<<<<<<<<<<<<<<
        # if True in np.isnan(rgbd_emb.cpu().numpy()):
        #         print("rgbd_emb is nan*******************8: {0}.".format(rgbd_emb))
        #         print("rgbd_emb shape: {0}.".format(rgbd_emb.shape))
        # print("code run here 4-end final up！")
        # print("rgbd_emb shape: {0}.".format(rgbd_emb.shape))
        # ###################### prediction stages #############################
        rgbd_segs = self.rgbd_seg_layer(rgbd_emb)
        
        pred_kp_ofs = self.kp_ofst_layer(rgbd_emb)
        pred_ctr_ofs = self.ctr_ofst_layer(rgbd_emb)
        # print("code run here 4.5-end prediction！")
        # print("rgbd_emb shape: {0}.".format(rgbd_emb.shape))
        # print("beforev prediction stages!!!!!!!!!!!!!!. rgb_emb_c: {0}. p_emb: {1}".format(pred_kp_ofs.shape, pred_ctr_ofs.shape))
        
        
            
        pred_ctr_ofs_score = torch.sigmoid(self.ctr_ofst_score_layer(rgbd_emb))
        pred_kp_ofs_score = torch.sigmoid(self.kp_ofst_score_layer(rgbd_emb))

        pred_kp_ofs = pred_kp_ofs.view(
            bs, self.n_kps, 3, -1
        ).permute(0, 1, 3, 2).contiguous() 
        pred_ctr_ofs = pred_ctr_ofs.view(
            bs, 1, 3, -1
        ).permute(0, 1, 3, 2).contiguous()
        #############################################
        pred_kp_ofs_score = pred_kp_ofs_score.view(
            bs, self.n_kps, 1, -1
        ).permute(0, 1, 3, 2).contiguous() 
        pred_ctr_ofs_score = pred_ctr_ofs_score.view(
            bs, 1, 1, -1
        ).permute(0, 1, 3, 2).contiguous()
        ############################################
        # print("code run here 5-end prediction！")
        # print("prediction stages!!!!!!!!!!!!!!. rgb_emb_c: {0}. p_emb: {1}".format(pred_kp_ofs.shape, pred_ctr_ofs.shape))
        # 归一化 变成单位向量
       
        # pred_kp_ofs = torch.div(pred_kp_ofs,torch.norm(pred_kp_ofs,2,3,True))
        # pred_ctr_ofs = torch.div(pred_ctr_ofs,torch.norm(pred_ctr_ofs,2,3,True))
        

        # return rgbd_seg, pred_kp_of, pred_ctr_of
        end_points['pred_rgbd_segs'] = rgbd_segs # [bs,N_pts,num_classes]
        end_points['pred_kp_ofs'] = pred_kp_ofs #[bs,n_kps,N_pts,3]
        end_points['pred_ctr_ofs'] = pred_ctr_ofs #[bs,1,N_pts,3]

        end_points['pred_ctr_ofs_score'] = pred_ctr_ofs_score #[bs,1,N_pts,1]
        end_points['pred_kp_ofs_score'] = pred_kp_ofs_score #[bs,n_kps,N_pts,1]
        # print("code run here 6-end network!")

        return end_points


# Copy from PVN3D: https://github.com/ethnhe/PVN3D
class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(64, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(64, 256, 1)

        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1)  # 128+ 512 + 1024 = 1664


def main():
    from common import ConfigRandLA
    rndla_cfg = ConfigRandLA
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    n_cls = 21
    model = FFB6D(n_cls, rndla_cfg.num_points, rndla_cfg)
    print(model)

    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )


if __name__ == "__main__":
    main()
