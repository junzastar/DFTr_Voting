from builtins import print
import math
import numpy as np
import torch
import sys
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))

class vector2Kps():
    def __init__(self):
        super(vector2Kps, self).__init__() 
    
    # [weighted vector-wise keypoints voting algorithm]
    def linear_decode_point(self, points_cur_obj, Pred_kp_vec, Pred_kp_vec_score, 
                        K = 30):
        r"""
            Parameters
            ----------
            points_cur_obj [num_points_obj,3]
            Pred_kp_vec [n_kps,num_points_obj,3]
            Pred_kp_vec_score [n_kps,num_points_obj,1]
            K topk biggest weight of Pred_kp_vec_score
            ----------
        """
        num_points_obj,_ = points_cur_obj.shape
        K_search = K
        if K_search > num_points_obj:
            K_search = num_points_obj
        Pred_kp_vec_score_K_value, Pred_kp_vec_score_K_idx = Pred_kp_vec_score.topk(K_search, dim=1, largest=True, sorted=True) # [n_kps,K,1]
        Pred_all_vec_score_K_idx = Pred_kp_vec_score_K_idx.repeat(1,1,3) #[n_kps, K,3]
        
        Pred_kp_vec = torch.gather(Pred_kp_vec,dim=1,index=Pred_all_vec_score_K_idx) # [n_kps, K, 3]
        points_cur_obj = points_cur_obj.view(1,-1,3).repeat(Pred_kp_vec.shape[0],1,1) #[n_kps,num_points_obj,3]
        points_cur_obj = torch.gather(points_cur_obj,dim=1,index=Pred_all_vec_score_K_idx) # [n_kps, K, 3]
        
        _, num_points_obj, _ = points_cur_obj.shape
        n_kps, _, _ = Pred_kp_vec.shape

        Kp_points = np.ones([n_kps,3]) 
        Pred_kp_vec = Pred_kp_vec.cpu().numpy()
        points_cur_obj = points_cur_obj.cpu().numpy()
        Pred_kp_vec_score_K_value = Pred_kp_vec_score_K_value.cpu().numpy() #[n_kps1, K, 1]
        Pred_kp_vec_score_K_value = Pred_kp_vec_score_K_value[:,:,:,np.newaxis] #[n_kps, K, 1, 1]
        if True in np.isnan(Pred_kp_vec_score_K_value):
            print("Pred_kp_vec_score_K_value is nan!!!!!")
        # solve
        nnt = Pred_kp_vec[:,:,:,np.newaxis] @ Pred_kp_vec[:, :, np.newaxis, :] #[n_kps, K, 3, 3]
        I = np.eye(3)
        R = np.sum(Pred_kp_vec_score_K_value * (I - nnt), axis=1) #[n_kps, 3, 3]
        q = np.sum(Pred_kp_vec_score_K_value * ((I - nnt) @ points_cur_obj[:,:,:,np.newaxis]), axis=1) #[n_kps, 3, 1]
        if (True in np.isnan(R)) or (True in np.isnan(q)):
            print("R or q is nan!!!!!")
        for kps_id in range(n_kps):
            Kp_points[kps_id,:] = np.linalg.lstsq(R[kps_id],q[kps_id],rcond=None)[0].T
        return Kp_points #[n_kps,3]

    

    ### [Another iterative optimization approach for keypoints position detection]
    ### Note: low detection accuracy in this version
    def decode_keypoint(self, points_cur_obj, Pred_kp_vec, Pred_kp_vec_score, 
                       K = 30, iterations_n = 50):
        r"""
            Parameters
            ----------
            points_cur_obj [num_points_obj,3]
            Pred_kp_vec [n_kps,num_points_obj,3]
            Pred_kp_vec_score [n_kps,num_points_obj,1]
            
            K topk biggest weight of Pred_kp_vec_score
        """

        num_points_obj,_ = points_cur_obj.shape
        K_search = K
        if K_search > num_points_obj:
            K_search = num_points_obj
        _, Pred_kp_vec_score_K_idx = Pred_kp_vec_score.topk(K_search, dim=1, largest=True, sorted=True) # [n_kps,K,1]
        _, Pred_ctr_vec_score_K_idx = Pred_ctr_vec_score.topk(K_search, dim=1, largest=True, sorted=True)#[1, K,1]
        
        Pred_all_vec_score_K_idx = Pred_kp_vec_score_K_idx.repeat(1,1,3) #[n_kps, K,3]

        Pred_kp_vec = torch.gather(Pred_kp_vec,dim=1,index=Pred_all_vec_score_K_idx) # [n_kps, K, 3]
        points_cur_obj = points_cur_obj.view(1,-1,3).repeat(Pred_kp_vec.shape[0],1,1) #[n_kps,num_points_obj,3]
        points_cur_obj = torch.gather(points_cur_obj,dim=1,index=Pred_all_vec_score_K_idx) # [n_kps, K, 3]-
        
        _, num_points_obj, _ = points_cur_obj.shape
        n_kps, _, _ = Pred_kp_vec.shape

        Kp_points = np.ones([n_kps,3]) #initialize
        iterations_n = iterations_n #iterations
        cur_cost=0.0 
        last_cost = 0.0 
        Pred_kp_vec = Pred_kp_vec.cpu().numpy()
        points_cur_obj = points_cur_obj.cpu().numpy()
        for kps_id in range(n_kps):
            for iterations in range(iterations_n):
                H_matrix = np.zeros([3,3]) # Hessian matrix
                b_vec = np.zeros([3,1]) # bias
                cur_cost = 0.0
                for points_id in range(num_points_obj):

                    ########################################### distacne ###########################
                    temp_a = (Kp_points[kps_id,1] - points_cur_obj[kps_id,points_id,1]) * Pred_kp_vec[kps_id,points_id,2] - (Kp_points[kps_id,2] - points_cur_obj[kps_id,points_id,2]) * Pred_kp_vec[kps_id,points_id,1]
                    temp_b = (Kp_points[kps_id,2] - points_cur_obj[kps_id,points_id,2]) * Pred_kp_vec[kps_id,points_id,0] - (Kp_points[kps_id,0] - points_cur_obj[kps_id,points_id,0]) * Pred_kp_vec[kps_id,points_id,2]
                    temp_c = (Kp_points[kps_id, 0] - points_cur_obj[kps_id,points_id, 0]) * Pred_kp_vec[
                        kps_id, points_id, 1] - (Kp_points[kps_id, 1] - points_cur_obj[kps_id,points_id, 1]) * \
                                Pred_kp_vec[kps_id, points_id, 0]
                    tem_num = temp_a**2 + temp_b**2 + temp_c**2
                    error = - np.sqrt(tem_num) #error function
                    J = np.ones([3,1]) #jacobian matrix

                    J[0] = - 0.5 * (1.0 / np.sqrt(tem_num + 1e-6)) * (2.0 * Pred_kp_vec[kps_id, points_id, 2]**2 * (Kp_points[kps_id, 0] - points_cur_obj[kps_id,points_id, 0]) -
                                            2.0 * Pred_kp_vec[kps_id, points_id, 0] * Pred_kp_vec[kps_id, points_id, 2] * (Kp_points[kps_id, 2] - points_cur_obj[kps_id,points_id, 2]) +
                                            2.0 * Pred_kp_vec[kps_id, points_id, 1]**2 * (Kp_points[kps_id, 0] - points_cur_obj[kps_id,points_id, 0]) -
                                            2.0 * Pred_kp_vec[kps_id, points_id, 0] * Pred_kp_vec[kps_id, points_id, 1] * (Kp_points[kps_id, 1] - points_cur_obj[kps_id,points_id, 1]))

                    J[1] = - 0.5 * (1.0 / np.sqrt(tem_num + 1e-6)) * (2.0 * Pred_kp_vec[kps_id, points_id, 2] ** 2 * (Kp_points[kps_id, 1] - points_cur_obj[kps_id,points_id, 1]) -
                                                2.0 * Pred_kp_vec[kps_id, points_id, 1] * Pred_kp_vec[kps_id, points_id, 2] * (Kp_points[kps_id, 2] - points_cur_obj[kps_id,points_id, 2]) +
                                                2.0 * Pred_kp_vec[kps_id, points_id, 0] ** 2 * (Kp_points[kps_id, 1] - points_cur_obj[kps_id,points_id, 1]) -
                                                2.0 * Pred_kp_vec[kps_id, points_id, 0] * Pred_kp_vec[kps_id, points_id, 1] * (Kp_points[kps_id, 0] - points_cur_obj[kps_id,points_id, 0]))

                    J[2] = - 0.5 * (1.0 / np.sqrt(tem_num +1e-6)) * (2.0 * Pred_kp_vec[kps_id, points_id, 1] ** 2 * (Kp_points[kps_id, 2] - points_cur_obj[kps_id,points_id, 2]) -
                                                2.0 * Pred_kp_vec[kps_id, points_id, 1] * Pred_kp_vec[kps_id, points_id, 2] * (Kp_points[kps_id, 1] - points_cur_obj[kps_id,points_id, 1]) +
                                                2.0 * Pred_kp_vec[kps_id, points_id, 0] ** 2 * (Kp_points[kps_id, 2] - points_cur_obj[kps_id,points_id, 2]) -
                                                2.0 * Pred_kp_vec[kps_id, points_id, 0] * Pred_kp_vec[kps_id, points_id, 2] * (Kp_points[kps_id, 0] - points_cur_obj[kps_id,points_id, 0]))
                    ##################################### distance #####################################
                    H_matrix += np.dot(J,J.transpose()) #3x3
                    b_vec += (-1.0) * error * J

                    cur_cost += error * error
                # solve
                dx = np.linalg.lstsq(H_matrix,b_vec,rcond=None)[0]
                    
                if np.isnan(dx[0].all()):
                    break
                if iterations > 0 and cur_cost >= last_cost:
                    break
                Kp_points[kps_id,:] = np.add(Kp_points[kps_id,:], dx.reshape(-1))
                last_cost = cur_cost
        return Kp_points #[n_kps,3]
