import numpy as np
import cv2

# https://github.com/IceTTTb/PlaneTR3D/
def evaluateDepths(predDepths, gtDepths, pred_mask, gt_mask, file_name = None, printInfo=False):
    """Evaluate depth reconstruction accuracy"""
    predDepths = predDepths.copy()
    gtDepths = gtDepths.copy()
    gt_mask = gt_mask.copy()
    gt_mask = cv2.resize((gt_mask<20).astype(np.uint8), (640, 480)) > 0 
    pred_mask = pred_mask.copy()
    pred_mask = cv2.resize((pred_mask<20).astype(np.uint8), (640, 480)) > 0 

    gtDepths = cv2.resize(gtDepths, (640, 480))
    predDepths = cv2.resize(predDepths, (640, 480))
    valid_mask = gtDepths > 1e-4

    valid_depth_mask = valid_mask * gt_mask * pred_mask
    if file_name != None:
        tmp_flag = cv2.imwrite("/home/jingjia/data/planerecon/nyuv2/nyuv2_plane/" + file_name + "_gt_pred_mask.png", (valid_depth_mask*255).astype(np.uint8))
        print(file_name, tmp_flag)

    gtDepths = gtDepths[valid_depth_mask] 
    predDepths = predDepths[valid_depth_mask]

    masks = gtDepths > 1e-4

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt(
        (pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
                1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass
    return np.array([rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3])