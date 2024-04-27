# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from PlaneRecTR import add_PlaneRecTR_config
from predictor import VisualizationDemo

from PlaneRecTR.utils.disp import visualizationBatch
import numpy as np
import torch


# constants
WINDOW_NAME = "PlaneRecTR demo"


def get_coordinate_map(K, device, h=192, w=256):
    K_inv = np.linalg.inv(np.array(K))

    K = torch.FloatTensor(K).to(device)
    K_inv = torch.FloatTensor(K_inv).to(device)


    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)

    return k_inv_dot_xy1


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_PlaneRecTR_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="PlaneRecTR demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/PlaneRecTRScanNetV1/PlaneRecTR_R50_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--fx",
        type=float,
        # default=0.5,
        help="camera K",
    )

    parser.add_argument(
        "--fy",
        type=float,
        # default=0.5,
        help="camera K",
    )

    parser.add_argument(
        "--ox",
        type=float,
        # default=0.5,
        help="camera K",
    )

    parser.add_argument(
        "--oy",
        type=float,
        # default=0.5,
        help="camera K",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = demo.run_on_image(img)
            sem_seg = predictions["sem_seg"].argmax(dim=0).cpu() # torch.Size([480, 640]) # sem_seg 21, 192, 256
            pred = np.array(sem_seg, dtype=np.int) # (480, 640)
            plane_depth = predictions["planes_depth"].cpu().numpy()
            K = [[args.fx, 0, args.ox],
            [0, args.fy, args.oy],
            [0, 0, 1]]
            vis_dicts = {
                        'image': img,
                        'segmentation': pred,
                        'depth_predplane': plane_depth,
                        'K_inv_dot_xy_1': get_coordinate_map(K, torch.device("cpu"), h=192, w=256).numpy(),
                    }
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )

            if args.output:
                if os.path.isdir(args.output):
                    # assert os.path.isdir(args.output), args.output
                    output_root = args.output
                    # out_filename = os.path.join(args.output, os.path.basename(path))
                    
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    output_root = os.path.split(args.output)[0]

                os.makedirs(output_root, exist_ok=True)
                
                visualizationBatch(root_path= output_root, idx="", info="", data_dict=vis_dicts, 
                                       num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, save_image=True, save_segmentation=True,
                                       save_depth=True, save_ply=True,save_cloud=False)
                # visualized_output.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()