import cv2
import numpy as np
import os
import json
from os.path import join as pjoin
import re

import tensorflow as tf  # 1.14.0
import numpy as np
import os
import argparse
import time


# os.environ['CUDA_VISIBLE_DEVICES']=''

# modified from https://github.com/art-programmer/PlaneNet
HEIGHT=192
WIDTH=256
NUM_PLANES = 20
NUM_THREADS = 4

scene_re = re.compile(r'scene\d+_\d+')
frame_re = re.compile(r'frame-\d+')

class RecordReaderAll:
    def __init__(self):
        return

    def getBatch(self, filename_queue, batchSize=1, min_after_dequeue=1000,
                 random=False, getLocal=False, getSegmentation=False, test=True):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'image_path': tf.FixedLenFeature([], tf.string),
                'num_planes': tf.FixedLenFeature([], tf.int64),
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                'segmentation_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'semantics_raw': tf.FixedLenFeature([], tf.string),                
                'boundary_raw': tf.FixedLenFeature([], tf.string),
                'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),                
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])

        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        normal = features['normal']
        normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])

        semantics = tf.decode_raw(features['semantics_raw'], tf.uint8)
        semantics = tf.cast(tf.reshape(semantics, [HEIGHT, WIDTH]), tf.int32)

        numPlanes = tf.cast(features['num_planes'], tf.int32)

        planes = features['plane']
        planes = tf.reshape(planes, [NUM_PLANES, 3])
        
        boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
        boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 2)), tf.float32)

        segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
        segmentation = tf.reshape(segmentation, [HEIGHT, WIDTH, 1])

        image_inp, plane_inp, depth_gt, normal_gt, semantics_gt, segmentation_gt, boundary_gt, num_planes_gt, image_path, info = \
            tf.train.batch([image, planes, depth, normal, semantics, segmentation, boundary, numPlanes, features['image_path'], features['info']], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=1)
        global_gt_dict = {'plane': plane_inp, 'depth': depth_gt, 'normal': normal_gt, 'semantics': semantics_gt,
                          'segmentation': segmentation_gt, 'boundary': boundary_gt, 'num_planes': num_planes_gt,
                          'image_path': image_path, 'info': info}
        return image_inp, global_gt_dict, {}

def get_plane_parameters(plane, plane_nums, segmentation):
    valid_region = segmentation != 20

    plane = plane[:plane_nums]

    tmp = plane[:, 1].copy()
    plane[:, 1] = -plane[:, 2]
    plane[:, 2] = tmp

    # convert plane from n * d to n / d
    plane_d = np.linalg.norm(plane, axis=1)
    # normalize
    plane /= plane_d.reshape(-1, 1)
    # n / d
    plane /= plane_d.reshape(-1, 1)

    h, w = segmentation.shape

    plane_parameters2 = np.ones((3, h, w))
    for i in range(plane_nums):
        plane_mask = segmentation == i
        plane_mask = plane_mask.astype(np.float32)
        cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
        plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

    # plane_instance parameter, padding zero to fix size
    plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
    return plane_parameters2, valid_region, plane_instance_parameter

def precompute_K_inv_dot_xy_1(h=192, w=256):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
            [0, focal_length, offset_y],
            [0, 0, 1]]

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
    depth_map[segmentation == 20] = gt_depth[segmentation == 20]
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
    
    gt_plane_depth = plane2depth(K_inv_dot_xy_1, plane_parameters, num_planes, gt_segmentation, gt_depth).reshape(1, 192, 256)

    return segments_info, gt_plane_depth, plane_instance_parameter[:num_planes]


def convert_frames_and_save_coco_json(input_folder, train_max_num = 50000, 
    val_max_num = 760, output_folder = None, 
    predict_center = True, train_depth_select = True, val_depth_select = False):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    K, K_inv_dot_xy_1, xy_map = precompute_K_inv_dot_xy_1()

    for split in ["train", "val"]:
    # for split in ["val"]:
        input_tfrecords_file = pjoin(input_folder, "planes_scannet_" + split + ".tfrecords")

        if split == 'train':
            file_list = open(pjoin(output_folder, 'train.txt'), 'w')
            output_root_dir = pjoin(output_folder, 'train')
            if not os.path.exists(output_root_dir):
                os.makedirs(output_root_dir)
            max_num = train_max_num
        elif split == 'val':
            file_list = open(pjoin(output_folder, 'val.txt'), 'w')
            output_root_dir = pjoin(output_folder, 'val')
            if not os.path.exists(output_root_dir):
                os.makedirs(output_root_dir)
            max_num = val_max_num
        else:
            print("unsupported data type")
            exit(-1)


        reader_train = RecordReaderAll()
        filename_queue_train = tf.train.string_input_producer([input_tfrecords_file], num_epochs=1)
        img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, batchSize=1, getLocal=True)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        annotations = []

        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(max_num):
                img, gt_dict = sess.run([img_inp_train, global_gt_dict_train])
                plane = gt_dict['plane'][0]
                depth = gt_dict['depth'][0]
                segmentation = gt_dict['segmentation'][0]
                num_planes = gt_dict['num_planes'][0].reshape([-1])
                image_path = gt_dict['image_path'][0].decode("utf-8") 
                
                gt_depth = depth.reshape(192, 256)
                gt_segmentation = segmentation.reshape((192, 256))
                segments_info, gt_plane_depth, plane_instance_parameters = process_frame_gts(plane, gt_depth, gt_segmentation, num_planes[0], xy_map, K_inv_dot_xy_1, predict_center)
                
                # print(split, locals()[split + '_depth_select'])
                if locals()[split + '_depth_select'] and (gt_plane_depth.min() < 0 or gt_plane_depth.max() > 6):
                    continue

                npz_file_name = pjoin(output_root_dir, '%d_d2.npz' % (i, ))
 
                np.savez(npz_file_name,
                        image=img[0], 
                        plane=plane_instance_parameters,
                        depth=gt_plane_depth, 
                        raw_depth = gt_depth,
                        segmentation=gt_segmentation, 
                        num_planes=num_planes,
                        )

                annotations.append({
                    "image_id": scene_re.search(image_path).group() + "_" + frame_re.search(image_path).group(),
                    "image_format": "CV2_BGR",
                    "width": int(WIDTH),
                    "height": int(HEIGHT),
                    "npz_file_name": npz_file_name,
                    "segments_info": segments_info,
                })

                file_list.write('%d_d2.npz\n' % (i, ))

                if i % 100 == 99: 
                    print(i)

        file_list.close()

        version = "1.0"
        data_created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        json_content = {
            "info": {
                "description": "scanet plane segmentation",
                "version": version,
                "data_created": data_created_time,
            },
            "camera": K,
            "categories": [{"id": 1, "name": "plane"}],
            "annotations": annotations,
        }
        output_json_file = pjoin(output_folder, "scannetv1_plane_len" + str(len(annotations)) + "_" + split + ".json")
        print("\nSaving the json file {}".format(output_json_file))
        with open(output_json_file, 'w') as f:
            json.dump(json_content, f, indent=4)

def dataset_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", default="/home/jingjia/data/planerecon/planenet/", help="path to input data")
    parser.add_argument("--output-folder", default="/home/jingjia/data/planerecon/", help="path to output data and json")
    
    return parser


if __name__ == '__main__':
    args = dataset_argument_parser().parse_args()
    convert_frames_and_save_coco_json(
        input_folder = args.input_folder, 
        train_max_num = 50000, 
        val_max_num = 760, 
        output_folder = os.path.join(args.output_folder, 'scannetv1_plane'), 
        predict_center = True, 
        train_depth_select = True, 
        val_depth_select = False)


        



