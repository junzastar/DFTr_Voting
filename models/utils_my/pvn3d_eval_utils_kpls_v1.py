from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import pickle as pkl
import concurrent.futures
import sys
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
from common import Config
from utils_my.basic_utils import Basic_Utils
from utils_my.meanshift_pytorch import MeanShiftTorch
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except Exception:
    from cv2 import imshow, waitKey
from utils_my.iteration_decode_kps import vector2Kps
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bop_toolkit.bop_toolkit_lib import pose_error, renderer
# from sixd_toolkit_old.pysixd import pose_error, inout

config = Config(ds_name='MP6D')
bs_utils = Basic_Utils(config)
cls_lst = config.ycb_cls_lst
try:
    config_lm = Config(ds_name="linemod")
    bs_utils_lm = Basic_Utils(config_lm)
except Exception as ex:
    print(ex)

# VSD parameters
vsd_delta = 1.5
vsd_tau =  [5]
vsd_cost = 'step' # 'step', 'tlinear'

mesh_pts_ply = []
# for obj_id in range(1,21):
#     path = '/home/rubbish/jun/dataset/MP6D_BOP/models_cad/obj_%06d.ply' % obj_id
#     ren.add_object(obj_id, path)
    # model_path = '/home/rubbish/jun/dataset/MP6D_BOP/models_cad/obj_%06d.ply' % cls_id
    # mesh_pts_ply.append(inout.load_ply(path))

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


# ###############################YCB Evaluation###############################
def cal_frame_poses(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
    gt_kps, gt_ctrs,
    pred_kp_ofs_score, pred_ctr_ofs_score, RTs,
    debug=False, kp_type='farthest', scenes_id=''
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    if debug:
        np.savetxt("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/{0}_cld_crop.txt", pcld.cpu().numpy())

    radius = 0.02
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    # Use center clustering filter to improve the predicted mask.
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
        ctrs = []
        for icls, cls_id in enumerate(pred_cls_ids):
            cls_msk = (mask == cls_id)
            ############### add by zj ######################
            Pred_ctr_vec = ctr_of[:, cls_msk,:] #[1,num_points_obj,3]
            points_cur_obj = pcld[cls_msk, :3]  # [num_points_obj,3]这个就是当前物体的所有点
            Pred_ctr_vec_score = pred_ctr_ofs_score[:, cls_msk,:] # [1, num_points_obj,1]
            
            it2kps = vector2Kps()
            # ctr = it2kps.decode_center_point(points_cur_obj,Pred_ctr_vec,Pred_ctr_vec_score,gt_ctr_ofs_cur,K=50,iterations_n=50)
            ctr = it2kps.linear_decode_center_point(points_cur_obj,
                                        Pred_ctr_vec,Pred_ctr_vec_score,K=n_pts) #[1,3]
            ctrs.append(ctr)
            ############### add by zj ######################
        try:
            ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
            n_ctrs, _ = ctrs.size()
            pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
            ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
            ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
            min_dis, min_idx = torch.min(ctr_dis, dim=1)
            msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
            new_msk = mask.clone()
            for cls_id in pred_cls_ids:
                if cls_id == 0:
                    break
                min_msk = min_dis < config.ycb_r_lst[cls_id-1] * 0.8
                update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
                new_msk[update_msk] = msk_closest_ctr[update_msk]
            mask = new_msk
        except Exception:
            pass

    # 3D keypoints voting and least squares fitting for pose parameters estimation.
    pred_pose_lst = []
    pred_kps_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3, :])
            pred_kps_lst.append(np.zeros((n_kps+1, 3)))
            continue

        ###### zj ############
        it2kps = vector2Kps()
        Pred_ctr_vec = ctr_of[:, cls_msk,:] #[1,num_points_obj,3]
        Pred_kp_vec = pred_kp_of[:, cls_msk,:] #[1,num_points_obj,3]
        points_cur_obj = pcld[cls_msk, :3]  # [num_points_obj,3]这个就是当前物体的所有点
        Pred_ctr_vec_score = pred_ctr_ofs_score[:, cls_msk,:] # [1, num_points_obj,1]
        Pred_kp_vec_score = pred_kp_ofs_score[:, cls_msk,:] # [1, num_points_obj,1]

        Pred_kp_vec = torch.div(Pred_kp_vec,torch.norm(Pred_kp_vec,2,2,True))
        Pred_ctr_vec = torch.div(Pred_ctr_vec,torch.norm(Pred_ctr_vec,2,2,True))

        Kp_points_kps = it2kps.linear_decode_point(points_cur_obj, Pred_kp_vec,Pred_kp_vec_score,
                                        K=80) #[8,3]
        Kp_points_ctr = it2kps.linear_decode_center_point(points_cur_obj,
                            Pred_ctr_vec,Pred_ctr_vec_score,K=80) #[1,3]
        Kp_points_kps = torch.from_numpy(np.array(Kp_points_kps).astype(np.float32))
        Kp_points_ctr = torch.from_numpy(np.array(Kp_points_ctr).astype(np.float32))
        for ikp in range(n_kps):
            cls_kps[cls_id, ikp, :] = Kp_points_kps[ikp,:]  # [3]
        cls_kps[cls_id, n_kps, :] = Kp_points_ctr[0,:]
        ###### zj ############

        # visualize
        # if debug:
        #     show_kp_img = np.zeros((480, 640, 3), np.uint8)
        #     kp_2ds = bs_utils.project_p3d(cls_kps[cls_id].cpu().numpy(), 1000.0)
        #     color = bs_utils.get_label_color(cls_id.item())
        #     show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
        #     imshow("kp: cls_id=%d" % cls_id, show_kp_img)
        #     waitKey(0)

        # Get mesh keypoint & center point in the object coordinate system.
        # If you use your own objects, check that you load them correctly.
        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1], kp_type=kp_type, ds_type="MP6D")
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1], ds_type="MP6D").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        pred_kpc = cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        pred_RT = best_fit_transform(mesh_kps, pred_kpc)
        pred_kps_lst.append(pred_kpc)
        pred_pose_lst.append(pred_RT)
        
        
        ### save yaml results ####
        # print("scenes_id: ", scenes_id)
        # path = "/home/rubbish/jun/FFB6D/ffb6d/results/{0}".format(scenes_id[0:4])
        # if not os.path.isdir(path):
        #     os.mkdir(path)
        # with open("/home/rubbish/jun/FFB6D/ffb6d/results/{0}/{1}_{2}.yml".format(scenes_id[0:4],scenes_id[5:],cls_id), 'w', encoding="utf-8") as RT_yml:
        #     cur_R = pred_RT[:, :3].flatten()
        #     cur_t = pred_RT[:, 3]
        #     RT_yml.write("run_time: 1.0" + "\n" + "ests:" + "\n" + \
        #             "- {score: 1.0,  + R: " + str(cur_R) + ", t: " + str(cur_t) + "}")
        #     RT_yml.close()
    


        if debug:
            print("pcld shape:", pcld.shape)
            print("Pred_kp_vec shape:", Pred_kp_vec.shape)
            print("Pred_ctr_vec shape:", Pred_ctr_vec.shape)
            print("pred_kpc shape:", pred_kpc.shape)
            print("mesh_kps shape:", mesh_kps.shape)
            print("pred_kp_of shape:", pred_kp_of.shape)
            print("ctr_of shape:", ctr_of.shape)
            with open("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/{0}_pred_point_kps_vec_from_dataset.txt".format(icls), 'a',encoding="utf-8") as kps:
                for k in range(points_cur_obj.shape[0]):
                    kps.write(str(points_cur_obj[k][0].cpu().numpy()) + "," + str(
                    points_cur_obj[k][1].cpu().numpy()) + "," + str(
                    points_cur_obj[k][2].cpu().numpy()) + ","+ str(
                    Pred_kp_vec[0][k][0].cpu().numpy()) + ","+str(
                    Pred_kp_vec[0][k][1].cpu().numpy()) +","+str(
                    Pred_kp_vec[0][k][2].cpu().numpy()) + "\n")
                kps.close()
            with open("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/{0}_pred_point_ctr_vec_from_dataset.txt".format(icls), 'a',encoding="utf-8") as ctr:
                for k in range(points_cur_obj.shape[0]):
                    ctr.write(str(points_cur_obj[k][0].cpu().numpy()) + "," + str(
                    points_cur_obj[k][1].cpu().numpy()) + "," + str(
                    points_cur_obj[k][2].cpu().numpy()) + ","+ str(
                    Pred_ctr_vec[0][k][0].cpu().numpy()) + ","+str(
                    Pred_ctr_vec[0][k][1].cpu().numpy()) +","+str(
                    Pred_ctr_vec[0][k][2].cpu().numpy()) + "\n")
                ctr.close()
            with open("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/{0}_pred_kps_from_dataset.txt".format(icls), 'a',encoding="utf-8") as kps:
                for k in range(9):
                    kps.write(str(pred_kpc[k][0]) + "," + str(
                    pred_kpc[k][1]) + "," + str(
                    pred_kpc[k][2]) + "\n")
                kps.close()
            with open("/home/rubbish/jun/FFB6D/ffb6d/train_log/MP6D/test/{0}_pred_ctr_from_dataset.txt".format(icls), 'a',encoding="utf-8") as mesh:
                for k in range(9):
                    mesh.write(str(mesh_kps[k][0]) + "," + str(
                    mesh_kps[k][1]) + "," + str(
                    mesh_kps[k][2]) + "\n")
                    kps.close()
                mesh.close()
    if debug:
        print("pred_cls_ids: ", pred_cls_ids)
        import sys
        sys.exit()        

    return (pred_cls_ids, pred_pose_lst, pred_kps_lst)


def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label,
    gt_kps, gt_ctrs, pred_kpc_lst, dpt_map
):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_kp_err = [list() for i in range(n_cls)]
    cls_vsd_err = [0 for i in range(n_cls)]
    cls_vsd_err_count = [0 for i in range(n_cls)]
    cls_vsd_err_count_TP = [0 for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        gt_kp = gt_kps[icls].contiguous().cpu().numpy()

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
            pred_kp = np.zeros(gt_kp.shape)
            R_e = pred_RT[:, :3].cpu().numpy()
            t_e = pred_RT[:, 3].cpu().numpy()
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            R_e = pred_RT[:, :3]
            t_e = pred_RT[:, 3]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        kp_err = np.linalg.norm(gt_kp-pred_kp, axis=1).mean()
        cls_kp_err[cls_id].append(kp_err)
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1], ds_type="MP6D").clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())


        ## VSD #####
        # R_e = pred_RT[:, :3]
        # t_e = pred_RT[:, 3]
        # gt_RT = RTs[icls]
        # try:
        R_g = gt_RT[:, :3]
        t_g = gt_RT[:, 3]
        K = config.intrinsic_matrix['MP6D_K']
        # mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1], ds_type="MP6D").clone()
        model_path = '/home/jiking/users/jun/datasets/MP6D_BOP/models_cad/obj_%06d.ply' % cls_id
        ren = renderer.create_renderer(640, 480, 'vispy', mode='depth')
        ren.add_object(cls_id.cpu().numpy()[0], model_path)
        # mesh_pts = inout.load_ply(model_path)
        # mesh_pts = mesh_pts_ply[cls_id-1]
        
        diameter = config.ycb_r_lst[cls_id-1] * 2.0
        # print("ycb_r_lst: ", config.ycb_r_lst)
        
        # print("cls_id: ", cls_id.cpu().numpy()[0])
        e = pose_error.vsd(R_e, t_e, R_g.cpu().numpy(), t_g.cpu().numpy(), dpt_map.cpu().numpy(), K, vsd_delta, vsd_tau,
                                            True, diameter, ren, cls_id.cpu().numpy()[0], vsd_cost)
        # e = pose_error.vsd(R_e, t_e, R_g.cpu().numpy(), t_g.cpu().numpy(), mesh_pts, dpt_map.cpu().numpy(), K, vsd_delta, vsd_tau, cost_type='tlinear')
        # print("VSD: " , e[0])
        
        # sys.exit()
        cls_vsd_err[cls_id] = e[0]
        cls_vsd_err_count[cls_id] += 1
        if e[0] < 0.3:
            cls_vsd_err_count_TP[cls_id] += 1
        import gc
        del e, ren
        gc.collect()
        # except RuntimeError as exception:
        #     print("WARNING: out of memory")
        #     if "out of memory" in str(exception):
        #         print("WARNING: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
    # sys.exit()
    return (cls_add_dis, cls_adds_dis, cls_kp_err, cls_vsd_err, cls_vsd_err_count, cls_vsd_err_count_TP)


def eval_one_frame_pose(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, gt_kps, gt_ctrs, kp_type, \
        pred_kp_ofs_score, pred_ctr_ofs_score, scenes_id, dpt_map = item

    # start_time = time.time()
    pred_cls_ids, pred_pose_lst, pred_kpc_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        gt_kps, gt_ctrs,
        pred_kp_ofs_score, pred_ctr_ofs_score, RTs, 
        kp_type=kp_type, scenes_id=scenes_id
    )
    # end_time = time.time()
    # print('Post-process second per frame=', (end_time-start_time))

    cls_add_dis, cls_adds_dis, cls_kp_err, cls_vsd_err, cls_vsd_err_count, cls_vsd_err_count_TP = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label, gt_kps, gt_ctrs,
        pred_kpc_lst, dpt_map
    )
    return (cls_add_dis, cls_adds_dis, pred_cls_ids, pred_pose_lst, cls_kp_err, cls_vsd_err, cls_vsd_err_count, cls_vsd_err_count_TP)

# ###############################End YCB Evaluation###############################


# ###############################LineMOD Evaluation###############################

def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.04
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        # visualize
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = bs_utils.project_p3d(
                cls_kps[cls_id].cpu().numpy(), 1000.0, K='linemod'
            )
            # print("cls_id = ", cls_id)
            # print("kp3d:", cls_kps[cls_id])
            # print("kp2d:", kp_2ds, "\n")
            color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
            show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)

        mesh_kps = bs_utils_lm.get_kps(obj_id, ds_type="linemod")
        if use_ctr:
            mesh_ctr = bs_utils_lm.get_ctr(obj_id, ds_type="linemod").reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)
    return pred_pose_lst


def eval_metric_lm(cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose_lm(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
        cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis)

# ###############################End LineMOD Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self, n_cls = 22):
        
        self.n_cls = n_cls
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]
        self.pred_vsd_errs = [0 for i in range(n_cls)]
        self.pred_vsd_errs_count = [0 for i in range(n_cls)]
        self.pred_vsd_errs_count_TP = [0 for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)

            ## vsd ##
            print("***************vsd<0.3_count: {0}. Recall: {1}.".format(self.pred_vsd_errs_count_TP[i], self.pred_vsd_errs_count_TP[i] / (self.pred_vsd_errs_count[i] + 1e-8)))
            print("***************overall_err: {0}. overall_count: {1}. mean_vsd: {2}\t".format(self.pred_vsd_errs[i], self.pred_vsd_errs_count[i], self.pred_vsd_errs[i] / (self.pred_vsd_errs_count[i] + 1e-8) ))
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))

    def cal_lm_add(self, obj_id, test_occ=False):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        cls_id = obj_id
        if (obj_id) in config_lm.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_lm.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_lm.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_lm.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lm.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            config_lm.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))

    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, 
        pred_kp_ofs_score, pred_ctr_ofs_score, dpt_map,#zj
        min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb', scenes_id=''
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        # if ds == "ycb":
        if ds == "ycb" or ds == "MP6D":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type, 
                pred_kp_ofs_score, pred_ctr_ofs_score, scenes_id, dpt_map
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        
        for res in map(eval_one_frame_pose, data_gen):
            # if ds == 'ycb':
            if ds == "ycb" or ds == "MP6D":
                cls_add_dis_lst, cls_adds_dis_lst, pred_cls_ids, pred_poses, pred_kp_errs , pred_vsd_err, cls_vsd_err_count, cls_vsd_err_count_TP= res
                self.pred_id2pose_lst.append(
                    {cid: pose for cid, pose in zip(pred_cls_ids, pred_poses)}
                )
                self.pred_kp_errs = self.merge_lst(
                    self.pred_kp_errs, pred_kp_errs
                )

                ## vsd ##
                self.pred_vsd_errs = self.merge_lst(
                    self.pred_vsd_errs, pred_vsd_err
                )
                self.pred_vsd_errs_count = self.merge_lst(
                    self.pred_vsd_errs_count, cls_vsd_err_count
                )
                self.pred_vsd_errs_count_TP = self.merge_lst(
                    self.pred_vsd_errs_count_TP, cls_vsd_err_count_TP
                )
            else:
                cls_add_dis_lst, cls_adds_dis_lst = res
            self.cls_add_dis = self.merge_lst(
                self.cls_add_dis, cls_add_dis_lst
            )
            self.cls_adds_dis = self.merge_lst(
                self.cls_adds_dis, cls_adds_dis_lst
            )

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

import queue
class BoundThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """
    对ThreadPoolExecutor 进行重写，给队列设置边界
    """
    def __init__(self, qsize: int = None, *args, **kwargs):
        super(BoundThreadPoolExecutor, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(qsize)

# vim: ts=4 sw=4 sts=4 expandtab
