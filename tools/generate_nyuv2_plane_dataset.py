
import cv2
import numpy as np
import os
import json
from os.path import join as pjoin

# import tensorflow as tf  # 1.14.0
import numpy as np
import os
import argparse
import time
import h5py
import scipy.io as sio


# os.environ['CUDA_VISIBLE_DEVICES']=''

# modified from https://github.com/art-programmer/PlaneNet
HEIGHT=192
WIDTH=256
NUM_PLANES = 20
NUM_THREADS = 4

def get_plane_parameters(plane, plane_nums, segmentation):
    valid_region = segmentation != NUM_PLANES

    plane = plane[:plane_nums]

    h, w = segmentation.shape

    plane_parameters2 = np.ones((3, h, w))
    for i in range(plane_nums):
        plane_mask = segmentation == i
        plane_mask = plane_mask.astype(np.float32)
        cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
        plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

    # plane_instance parameter, padding zero to fix size
    plane_instance_parameter = np.concatenate((plane, np.zeros((NUM_PLANES - plane.shape[0], 3))), axis=0)
    return plane_parameters2, valid_region, plane_instance_parameter

def precompute_K_inv_dot_xy_1(h=192, w=256):
    # K from toolbox_nyu_depth_v2
    # % RGB Intrinsic Parameters 
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    # % Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02

    K = [[fx_rgb, 0, cx_rgb],
            [0, fy_rgb, cy_rgb],
            [0, 0, 1]]

    # focal_length = 5.8262448167737955e+02
    # offset_x = 3.1304475870804731e+02
    # offset_y = 2.3844389626620386e+02

    # K = [[focal_length, 0, offset_x],
    #         [0, focal_length, offset_y],
    #         [0, 0, 1]]
    

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))
    xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv,
                            np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]
            xy_map[0, y, x] = float(x) / w
            xy_map[1, y, x] = float(y) / h

    # precompute to speed up processing
    return K, K_inv_dot_xy_1, xy_map

def plane2depth(K_inv_dot_xy_1, plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):

    depth_map = 1. / np.sum(K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    depth_map = depth_map.reshape(h, w)

    # replace non planer region depth using sensor depth map
    depth_map[segmentation == NUM_PLANES] = gt_depth[segmentation == NUM_PLANES]
    return depth_map

def cal_mask_bbox(mask):
    # area = np.sum(mask) # segment area computation

    # bbox computation for a segment
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    bbox = [int(x), int(y), int(width), int(height)]

    return bbox


def process_frame_gts(plane, gt_depth, gt_segmentation, num_planes, xy_map, K_inv_dot_xy_1, predict_center):
    """generate segments_info

    Args:
        plane (_type_): size 20x3
        gt_depth (_type_): size 196x256
        gt_segmentation (_type_): plane id from 0, 20 for non-label/non-plane
        num_planes (integer): real num of planes < 20
        xy_map (_type_): _description_
        K_inv_dot_xy_1 (_type_): _description_
        predict_center (bool): _description_
    """

    plane = plane.copy()
    plane_parameters, valid_region, plane_instance_parameter = \
        get_plane_parameters(plane, num_planes, gt_segmentation)  # plane [20, 3], gt_segmentation [192, 256], plane_parameters [3, 192, 256], plane_instance_parameter [20,3]

    segments_info = []
    for i in range(num_planes):
        plane_mask = gt_segmentation == i
        pixel_num = plane_mask.sum()
        # plane_x, plane_y = (0, 0)
        bbox = cal_mask_bbox(plane_mask)
        item = {
            "id": i,
            # "color": color,
            "bbox": bbox,
            "bbox_mode": 1, #<BoxMode.XYWH_ABS: 1>
            "iscrowd": 0, # 0 for polygons format, 1 for RLE format
            "area": int(pixel_num),
        }

        if predict_center:
            plane_mask = plane_mask.astype(np.float)
            x_map = xy_map[0] * plane_mask
            y_map = xy_map[1] * plane_mask
            x_sum = x_map.sum()
            y_sum = y_map.sum()
            plane_x = x_sum / pixel_num
            plane_y = y_sum / pixel_num
            item["center"] = [plane_x, plane_y]

        segments_info.append(item)

            

    # since some depth is missing, we use plane to recover those depth following PlaneNet
    
    gt_plane_depth = plane2depth(K_inv_dot_xy_1, plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)
    
    return segments_info, gt_plane_depth, plane_instance_parameter[:num_planes]

def convert_frames_and_save_coco_json(input_folder, 
    output_folder = "/data/jingjia/nyuv2/", 
    predict_center = True, depth_select = False):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    K, K_inv_dot_xy_1, xy_map = precompute_K_inv_dot_xy_1()

    output_root_dir = output_folder
    file_list = open(pjoin(output_folder, 'test.txt'), 'w')


    annotations = []

    max_num_planes = 0

    mat_data = h5py.File(os.path.join(input_folder, 'nyu_depth_v2_labeled.mat'))
    split = sio.loadmat(os.path.join(input_folder, 'splits.mat'))
    test_indices = split['testNdxs'].reshape(-1) - 1 # 654

    # for i in range(test_max_num):
    for i in test_indices:
        img_file_path = pjoin(input_folder, 'nyu_depth_v2_plane_gt', 'plane_instance_%d.png' % (i+1))
        input_npz_file_path = pjoin(input_folder, 'nyu_depth_v2_plane_gt', 'plane_instance_%d.npz' % (i+1))

        img = cv2.imread(img_file_path)[:HEIGHT,:WIDTH,:] # (192, 256, 3) #!crop the white edges!
        img_tmp = mat_data['images'][i].transpose((2, 1, 0)).astype(np.uint8) # (480, 640, 3) BGR?
        img_tmp = cv2.resize(img_tmp, (WIDTH, HEIGHT))
        # cv2.im
        # assert (img==img_tmp).all(), "(img==img_tmp).all()"

        gt_depth = mat_data['depths'][i].transpose((1, 0)).astype(np.float32)

        # ! crop and resize
        gt_depth = gt_depth[44:471, 40:601]
        gt_depth = cv2.resize(gt_depth, (640, 480))

        gt_depth = cv2.resize(gt_depth, (WIDTH, HEIGHT))
        input_npz = np.load(input_npz_file_path)

        
        plane = input_npz['plane_param'] # [n, 3]
        segmentation = input_npz['plane_instance']
        # plane id from 0, NUM_PLANES for non-label/non-plane regions
        non_label_mask = segmentation==0
        segmentation -= 1
        segmentation[non_label_mask] = NUM_PLANES

        num_planes = np.array([plane.shape[0]])
        assert num_planes == (np.unique(segmentation).shape[0]-1), "num of plane params != num of plane instances in segmentation"
        if num_planes > max_num_planes:
            max_num_planes = num_planes
        

        # gt_depth = np.zeros((HEIGHT, WIDTH)) # TODO: add real gt_depths
        gt_segmentation = segmentation.reshape((HEIGHT, WIDTH))
        segments_info, gt_plane_depth, plane_instance_parameters = process_frame_gts(plane, gt_depth, gt_segmentation, num_planes[0], xy_map, K_inv_dot_xy_1, predict_center)
        
        if depth_select and (gt_plane_depth.min() < 0 or gt_plane_depth.max() > 10): #  scannetv1 > 6
            continue

        
        npz_file_name = pjoin(output_root_dir, '%d_d2.npz' % i)
        np.savez(npz_file_name,
                image=img,  #! scannetv1_planeTR img[0]
                raw_image=img_tmp,
                plane=plane_instance_parameters,  # num_planes, 3
                depth = gt_plane_depth, # (1, HEIGHT, WIDTH)
                raw_depth = gt_depth, # (HEIGHT, WIDTH)
                segmentation=gt_segmentation,  # NUM_PLANES for non-plane, 0-(NUM_PLANES-1) for planes
                num_planes=num_planes,
                )


        annotations.append({
            "image_id": str(i) + '_d2',
            "image_format": "CV2_BGR",
            "width": int(WIDTH),
            "height": int(HEIGHT),
            "npz_file_name": npz_file_name,
            "segments_info": segments_info,
        })

        file_list.write('%d_d2.npz\n' % (i, ))

        if i % 100 == 99: 
            print(i)

    print("max_num_planes: ", max_num_planes)

    file_list.close()

    version = "1.0" if not depth_select else "2.0"
    data_created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    json_content = {
        "info": {
            "description": "nyuv2 plane segmentation, depth is cropped, plane is true from planeAE",
            "version": version,
            "data_created": data_created_time,
        },
        "camera": K,
        "categories": [{"id": 1, "name": "plane"}],
        "annotations": annotations,
    }
    output_json_file = pjoin(output_folder, "nyuv2_plane_len" + str(len(annotations)) + "_test" + ".json")
    print("\nSaving the json file {}".format(output_json_file))
    with open(output_json_file, 'w') as f:
        json.dump(json_content, f, indent=4)

def dataset_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default="/home/jingjia/data/planerecon/nyuv2/", help="path to input data")
    parser.add_argument("--output-folder", default="/home/jingjia/data/planerecon/", help="path to output data and json")
    
    return parser


if __name__ == '__main__':
    args = dataset_argument_parser().parse_args()
    
    convert_frames_and_save_coco_json(
        input_folder = args.input_folder, 
        output_folder = os.path.join(args.output_folder, 'nyuv2_plane'), 
        predict_center = True, 
        depth_select = False)