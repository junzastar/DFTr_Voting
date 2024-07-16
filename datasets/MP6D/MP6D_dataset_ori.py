#!/usr/bin/env python3
from ast import arg
import sys
from copy import copy
from builtins import float, int, print
from distutils.log import ERROR
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import numpy.ma as ma
from common import Config
import pickle as pkl
from utils_my.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP


config = Config(ds_name='MP6D')
bs_utils = Basic_Utils(config)


class Dataset():

    def __init__(self, dataset_name, DEBUG=False):
        self.dataset_name = dataset_name
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = bs_utils.read_lines(config.ycb_cls_lst_p)
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            self.path = 'datasets/MP6D/dataset_config/train_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
            self.minibatch_per_epoch = len(self.all_lst) // config.mini_batch_size
            self.real_lst = []
            self.syn_lst = []
            for item in self.all_lst:
                if item[:5] == 'data/':
                    self.real_lst.append(item)
                else:
                    self.syn_lst.append(item)
        else:
            self.pp_data = None
            self.add_noise = False
            self.path = 'datasets/MP6D/dataset_config/test_data_list.txt'
            self.all_lst = bs_utils.read_lines(self.path)
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))
        self.root = config.ycb_root
        self.sym_cls_ids = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.image_H = 480
        self.image_W = 640
        self.index = 1

    def real_syn_gen(self):
        if self.rng.rand() > 0.8:
            n = len(self.real_lst)
            idx = self.rng.randint(0, n)
            item = self.real_lst[idx]
        else:
            n = len(self.syn_lst)
            idx = self.rng.randint(0, n)
            item = self.syn_lst[idx]
        return item

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K, xmap, ymap):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (ymap - K[0][2]) * dpt / K[0][0]
        col = (xmap - K[1][2]) * dpt / K[1][1]
       
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    

    

    def get_item(self, item_name):
        with Image.open(os.path.join(self.root, item_name+'-depth.png')) as di:
            dpt_um = np.array(di)
            dpt_m = dpt_um / 1000.0
        
        with Image.open(os.path.join(self.root, item_name+'-label.png')) as li:
            labels = np.array(li)
        rgb_labels = labels.copy()
        meta = scio.loadmat(os.path.join(self.root, item_name+'-meta.mat'))
        # if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
        #     K = config.intrinsic_matrix['MP6D_K2']
        # else:
        #     K = config.intrinsic_matrix['MP6D_K1']
        K = config.intrinsic_matrix['MP6D_K']
        # print("code here 1")
        # print("K: {0}".format(K))

        with Image.open(os.path.join(self.root, item_name+'-color.png')) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
            rgb_copy = rgb.copy()
        rnd_typ = 'syn' if 'syn' in item_name else 'real'
        cam_scale = meta['factor_depth'].astype(np.float32)[0][0]
        # print("cam_scale: ", cam_scale)
        msk_dp = dpt_um > 1e-6
        # print("code here 2")
        # print("msk_dp non zero: {0}".format(np.count_nonzero(msk_dp)))

        if self.add_noise and rnd_typ == 'syn':
            rgb = self.rgb_add_noise(rgb)
            #rgb, dpt_um = self.add_real_back(rgb, rgb_labels, dpt_um, msk_dp) # MP6D 合成数据有真实背景
            # print("dpt_um non zero noise: {0}".format(np.count_nonzero(dpt_um)))
            if self.rng.rand() > 0.8:
                rgb = self.rgb_add_noise(rgb)
            rgb_copy = rgb.copy()


        dpt_mm = (dpt_um.copy()).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        # print("code here 3")
        #print("msk_dp: {0}".format(msk_dp))

        if self.debug:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            # imshow("nrm_map", show_nrm_map)
            cv2.imwrite("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/nrm_map.png",show_nrm_map)
        
        dpt_m = dpt_m.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K, self.xmap, self.ymap).reshape(-1, 3)
        

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]
        # print("code here 5")

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]
        

        cld = dpt_xyz.reshape(-1, 3)[choose, :] # 从深度图转出来的点云单位也是毫米，需要转成米为单位
        

        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        
        labels_pt = labels.flatten()[choose]
        # labels_pt = inst_label.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, cls_id_lst, meta
        )
        

        h, w = rgb_labels.shape
        
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
        


        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            
            cld = sub_pts

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
      

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
            scenes_id=item_name[5:]
        )
        item_dict.update(inputs)
        if self.debug:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, cls_id_lst, meta):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            r = meta['poses'][:, :, i][:, 0:3]
            t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            t = t / 1000.0 # modified by zj （M）
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = bs_utils.get_ctr(self.cls_lst[cls_id-1]).copy()[:, None]
            ctr = np.dot(ctr.T, r.T) + t[:, 0]  # modified by zj 转到相机坐标系（M）
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            # pred_cls_ids = np.unique(labels[labels > 0])
            # print("pred_cls_ids: ", pred_cls_ids)
            # print("cls_id_lst: ", cls_id_lst)
            # print("labels: ", len(labels))
            # print("labels: ", labels.shape)


            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([cls_id])
            # print("before norm kp_targ_ofst ", target_offset)
            # print("before norm ctr_targ_ofst ", ctr_targ_ofst)

            key_kpts = ''
            if config.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(config.n_keypoints)
            kps = bs_utils.get_kps(
                self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='MP6D'
            ).copy()
            # print("kps: {0}".format(kps))
            kps = np.dot(kps, r.T) + t[:, 0] # modified by zj 转到相机坐标系 （M）
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
            ### 归一化
            ###
            # print("kp_targ_ofst shape ", kp_targ_ofst[msk_idx, :, :].shape)
            # kp_targ_ofst[msk_idx, :, :] = kp_targ_ofst[msk_idx, :, :] / np.linalg.norm(kp_targ_ofst[msk_idx, :, :], 2, 2, True)
            # ctr_targ_ofst[msk_idx, :] = ctr_targ_ofst[msk_idx, :] / np.linalg.norm(ctr_targ_ofst[msk_idx, :], 2, 1, True)
            # print("test! kp_targ_ofst: ", kp_targ_ofst[msk_idx, :, :])
            # print("test! ctr_targ_ofst: ", ctr_targ_ofst[msk_idx, :, :])
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)

# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
# border_list = [-1, 120, 240, 360, 480, 600, 720]
border_list = [-1, 80, 160, 240, 320, 400, 480, 560, 640, 720]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def view_labels(self,rgb_chw, cld_cn, labels, K=config.intrinsic_matrix['MP6D_K']):
    rgb_hwc = np.transpose(rgb_chw, (0,1,2)).astype("uint8").copy()
    cld_nc = np.transpose(cld_cn, (0,1)).copy()
    p2ds = bs_utils.project_p3d(cld_nc, 1.0, K).astype(np.int32)
    # print("p2ds.size: {0}".format(p2ds.shape))
    # print("k: {0}".format(K))
    # print("rgb.size: {0}".format(rgb_hwc.shape))
    # print("label shape: {0}".format(labels.shape))
    labels = np.array(labels)
    labels = labels.flatten()
    # labels = labels.squeeze()
    
    colors = []
    h, w = rgb_hwc.shape[0], rgb_hwc.shape[1]
    rgb_hwc = np.zeros((h, w, 3), "uint8")
    color_lst = [(0,0,0)]
    for lb in labels:
        # print("lb : {0}".format(lb))
        if int(lb) == 0:
            c = (0, 0, 0)
        else:
            c = (100,25,36)
        colors.append(c)
    show = bs_utils.draw_p2ds(rgb_hwc, p2ds, 2, color=colors)
    return show

if __name__ == "__main__":
    # main()
    ds = {}
    ds['train'] = Dataset('train', DEBUG=False)
    ds['test'] = Dataset('test', DEBUG=False)
    idx = dict(
        train=0,
        val=0,
        test=0
    )
    
    i = 1
    max_num = 0
    # while True:
    for index_my in range(0,100):
        # for cat in ['val', 'test']:
        # for cat in ['train']:
        for cat in ['test']:
            datum = ds[cat].__getitem__(idx[cat])
            idx[cat] += 1
            rgb = datum['rgb'].transpose(1, 2, 0)[...,::-1].copy()# [...,::-1].copy()
            pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
            # norm_pcd = datum['cld_rgb_nrm'][3:6, :].transpose(1, 0).copy() # [192*192,3]
            obj_idx = datum['cls_ids']
            # obj_idx2 = datum['obj_idx']
            choose = datum['choose']
            kp_targ_ofst = datum['kp_targ_ofst'].transpose(1,0,2)
            ctr_targ_ofst = datum['ctr_targ_ofst']
            kp3d = datum['kp_3ds'][0]
            ctr3d = datum['ctr_3ds'][0]
            scene_id = datum['scenes_id']
            # print("datum['ctr_3ds']: {0}".format(datum['ctr_3ds']))
            # model_cld = datum['model_cld'].transpose(1, 0).copy()
            # model_cld_norm = datum['model_cld_norm'].copy()
            max_num = torch.max(torch.from_numpy(datum['choose']))
# vim: ts=4 sw=4 sts=4 expandtab
