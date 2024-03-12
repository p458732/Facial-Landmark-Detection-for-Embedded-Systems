import torch
import numpy as np

class MSE:
    def __init__(self):
        pass
    def __repr__(self):
        return "MSE()"
    def test(self, label_pd, label_gt):
        assert label_pd.shape == label_gt.shape, 'the nme_right_index is not list.'
        mse_list = []
        label_pd = label_pd.data.cpu().numpy()
        label_gt = label_gt.data.cpu().numpy()
        for i in range(label_gt.shape[0]):
            landmarks_gt = label_gt[i]
            landmarks_pv = label_pd[i]
            
            mse_list.append(np.linalg.norm(landmarks_gt - landmarks_pv))
            
        return mse_list

        for i in range(label_gt.shape[0]):
            landmarks_gt = label_gt[i]
            landmarks_pv = label_pd[i]
            if isinstance(self.nme_right_index, list):
                norm_distance = self.get_norm_distance(landmarks_gt)
            elif isinstance(self.nme_right_index, int):
                norm_distance = np.linalg.norm(landmarks_gt[self.nme_left_index] - landmarks_gt[self.nme_right_index])
            else:
                raise NotImplementedError
            landmarks_delta = landmarks_pv - landmarks_gt
            nme = (np.linalg.norm(landmarks_delta, axis=1) / norm_distance).mean()
            nme_list.append(nme)
            # sum_nme += nme
            # total_cnt += 1
        return nme_list
