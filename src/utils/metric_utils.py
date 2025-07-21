import numpy as np
import torch



def ACC(y, pred, lat_weight):
    '''
    y, pred are long-term-mean-subtracted values
    '''
    x = np.sum(y*pred*lat_weight)
    y = np.sqrt(np.sum(y*y*lat_weight) * np.sum(pred*pred*lat_weight))
    return x/y


def RMSE(y, pred, lat_weight):
    x = np.sum((y-pred)**2 * lat_weight)
    size = y.shape
    rmse = np.sqrt(x/size[-1]/size[-2]/size[0])
    return rmse


def stat_ts_cnt(gt, pred, threshold=0.1, axis=(-1,-2)):
    tp = ((gt>=threshold) * (pred>=threshold)).sum(axis)
    tn = ((gt<threshold) * (pred<threshold)).sum(axis)
    fp = ((gt<threshold) * (pred>=threshold)).sum(axis)
    fn = ((gt>=threshold) * (pred<threshold)).sum(axis)
    cnts = np.stack((tp, tn, fp, fn), dtype=np.float32)
    return cnts


def csi_pod(cnts):
    scores = {}
    scores["csi"] = cnts["tp"] / (cnts["tp"] + cnts["fp"] + cnts["fn"])
    scores["pod"] = cnts["tp"] / (cnts["tp"] + cnts["fn"])
    scores["far"] = cnts["fp"] / (cnts["fp"] + cnts["tp"])
    scores["bias"] = (cnts["tp"] + cnts["fp"]) / (cnts["tp"] + cnts["fn"])
    return scores


def prob_match(ensemble_members):
    n, height, width = ensemble_members.shape
    # Calculate the EnMax (element-wise maximum across the ensemble members)
    enmean = np.mean(ensemble_members, axis=0)
    enmean = enmean.flatten()
    sort_indices = np.argsort(enmean, axis=None, kind="quiksort")
    enmean_sorted = enmean[sort_indices]
    ensemble_members = ensemble_members.flatten()
    sort_ens_indices = np.argsort(ensemble_members, axis=None, kind="quiksort")
    msk = enmean_sorted<1e-9
    enmean_sorted = ensemble_members[sort_ens_indices[n-1::n]]
    enmean_sorted[msk] = 0
    pm_result = enmean_sorted[np.argsort(sort_indices, axis=None, kind="quiksort")]
    pm_result = pm_result.reshape(height, width)
    return pm_result


def pm_enmax(ensemble_members):
    """
    Probability Matching using Ensemble Maximum (EnMax) method.
    
    Parameters:
    ensemble_members (np.ndarray): A 3D array of shape (n_members, height, width)
    
    Returns:
    np.ndarray: The PM EnMax result of shape (height, width)
    """
    # Ensure the input is a 3D or 4D array
    assert ensemble_members.ndim in [3,4], ValueError("Input must be a 3D or 4D array")
    if isinstance(ensemble_members, torch.Tensor):
        ensemble_members = ensemble_members.detach().cpu().numpy()
    if ensemble_members.ndim==4:
        b = ensemble_members.shape[0]
        pms = []
        for i in range(b):
            pms.append(prob_match(ensemble_members[i]))
        return np.stack(pms, axis=0)
    else:
        return prob_match(ensemble_members)
    

