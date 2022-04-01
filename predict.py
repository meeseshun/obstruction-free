#! /usr/bin/python3

import cv2
import numpy as np
import networkx as nx
import os
import sys
import shutil
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from typing import Union
from inpaint import inpaint

DEBUG_IMAGES_DIR: str = 'img_dbg'
INPUT_IMAGE_PATH: str = 'input.png'
MASK_IMAGE_PATH: str = 'mask.png'


VERBOSE: bool = False
MORE_VERBOSE: bool = False
SAVE_DEBUG_IMAGES: bool = False
SKIP_HOMOGRAPHY_CHECK: bool = False
INVERSE_MASK: bool = False
NO_INPAINTING: bool = False
USE_GPU: bool = False


def log_debug(message: str) -> None:
    if VERBOSE:
        print(message, file=sys.stderr)


def log_warn(message: str) -> None:
    print('Warn: '+message, file=sys.stderr)


def log_error(message: str) -> None:
    assert False, message


def load_image(img_file: str) -> np.ndarray:
    assert os.path.isfile(img_file), f'No input file: {img_file}'
    img = cv2.imread(str(img_file))
    assert img is not None, f'load failed: {img_file}'
    return img


def save_image(img_file: Union[str, Path], img: np.ndarray) -> None:
    assert cv2.imwrite(str(img_file), img), f'save failed: {img_file}'
    assert os.path.isfile(img_file), f'No output file: {img_file}'


def save_debug_images(debug_images: list, file_prefix: str, extension: str = 'jpg') -> None:
    d = Path(DEBUG_IMAGES_DIR)
    os.makedirs(d, exist_ok=True)
    for i, debug_image in enumerate(debug_images):
        save_image(d / f'{file_prefix}_{i:02}.{extension}', debug_image)


def detect_keypoints_and_compute_descriptors(imgs) -> tuple:
    akaze = cv2.AKAZE_create()

    kp, des = [], []
    for img in tqdm(imgs, disable=not MORE_VERBOSE):
        k, d = akaze.detectAndCompute(img, None)
        kp.append(k)
        des.append(d)
    return kp, des


def find_all_homographies(kp: list, des: list) -> list:
    assert len(kp) == len(des), f'Length mismatch: {len(kp)} != {len(des)}'

    N = len(kp)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # TODO precalc pts

    H = [[None for _ in range(N)] for _ in range(N)]
    with tqdm(total=N*(N-1)//2, disable=not MORE_VERBOSE) as pbar:
        for j in range(N):
            for i in range(j):
                matches = bf.match(des[i], des[j])

                src_pts = np.float32(
                    [kp[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H[i][j], _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

                pbar.update(1)
    return H


def is_valid_homography(m: np.ndarray, shape: tuple) -> bool:
    def dist(a, b):
        return np.linalg.norm(a - b)

    assert m.shape == (3, 3), \
        f'should be a homography matrix (3x3): {m.shape}'

    h, w = shape[:2]
    thresh_dist = min(h, w)/5  # no basis

    v_ul = dist(m.dot(np.float32([0, 0, 1]))[:2], (0, 0)) < thresh_dist
    v_ur = dist(m.dot(np.float32([0, w, 1]))[:2], (0, w)) < thresh_dist
    v_ll = dist(m.dot(np.float32([h, 0, 1]))[:2], (h, 0)) < thresh_dist
    v_lr = dist(m.dot(np.float32([h, w, 1]))[:2], (h, w)) < thresh_dist

    return all([v_ul, v_ur, v_ll, v_lr])


def remove_invalid_homographies(H: list, shape: tuple) -> int:
    count = 0
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] is not None:
                if not is_valid_homography(H[i][j], shape):
                    H[i][j] = None
                    count += 1
    return count


def add_inverse_homographies(H: list) -> None:
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] is not None:
                H[j][i] = np.linalg.inv(H[i][j])


def H2G(H: list) -> nx.Graph:
    G = nx.Graph()

    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] is not None:
                G.add_edge(i, j)

    return G


def align_images(imgs: list, H: list) -> list:
    '''align images from homography matrixes'''

    # add alpha channel to all images
    # imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    #         for img in imgs]

    h, w = imgs[0].shape[:2]

    G = H2G(H)

    log_debug('dot graph')
    try:
        log_debug(nx.drawing.nx_agraph.to_agraph(G))
    except ImportError:
        log_debug('showing dot graph requires pygraphviz')

    try:
        shortest_path_to_base = nx.shortest_path(G, target=0)
    except nx.exception.NodeNotFound:
        log_error('Too few matching points to align')

    aligned_images = [imgs[0]]

    for i, img in enumerate(imgs[1:], 1):
        try:
            path = shortest_path_to_base[i]
        except KeyError:
            log_warn(f'No path to align {i}th image, ignoring...')
            continue
        H_base = np.eye(3)
        for j in range(len(path)-1):
            H_base = H_base.dot(H[path[j+1]][path[j]])

        aligned_image = cv2.warpPerspective(img, H_base, (w, h))

        aligned_images.append(aligned_image)

    return aligned_images


def make_mask(img_sub: np.ndarray) -> np.ndarray:
    _, img_sub_bin = cv2.threshold(img_sub, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), dtype=np.uint8)

    img_open = cv2.morphologyEx(img_sub_bin, cv2.MORPH_OPEN, kernel)

    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)

    mask_bin = cv2.bitwise_not(img_close)

    mask = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)

    return (img_close, mask_bin), mask


def make_masks(imgs: list) -> list:
    N = len(imgs)

    # assert N >= 3, 'Need 3 or more images'

    vs = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)[:, :, 2] for img in imgs]
    vs_norm = [cv2.equalizeHist(v) for v in vs]

    img_mean = np.mean(vs_norm, axis=0).astype(np.uint8)

    masks = []

    c = 0
    for v in vs_norm:
        s = cv2.subtract(v, img_mean)

        (mask_bin_not, mask_bin), _ = make_mask(s)

        _, std_f = cv2.meanStdDev(v, mask=mask_bin_not)
        _, std_b = cv2.meanStdDev(v, mask=mask_bin)

        c += std_f > std_b

    flag = c < (N-1)/2

    if INVERSE_MASK:
        flag = not flag

    for v in vs_norm:
        if flag:
            s = cv2.subtract(v, img_mean)
        else:
            s = cv2.subtract(img_mean, v)

        _, mask = make_mask(s)

        masks.append(mask)

    return masks


def compose(imgs: list, masks: list) -> tuple:
    N = min(len(imgs), len(masks))

    assert imgs[0].shape[2] == masks[0].shape[2], 'channel mismatch'

    img_add = np.zeros_like(imgs[0], dtype=np.uint8)
    mask_or = np.zeros_like(imgs[0], dtype=np.uint8)
    for i in range(N):
        mask_and = np.full_like(imgs[0], 255, dtype=np.uint8)
        for j in range(i):
            mask_and = cv2.bitwise_and(mask_and, cv2.bitwise_not(masks[j]))
        mask_and = cv2.bitwise_and(mask_and, masks[i])
        img_add = cv2.add(cv2.bitwise_and(img_add, mask_or),
                          cv2.bitwise_and(imgs[i], mask_and))
        mask_or = cv2.bitwise_or(mask_or, masks[i])

    if SAVE_DEBUG_IMAGES:
        save_image(Path(DEBUG_IMAGES_DIR) / 'compose.jpg', img_add)

    mask_not = np.full_like(masks[0], 255, dtype=np.uint8)
    for mask in masks:
        mask_not = cv2.bitwise_and(mask_not, cv2.bitwise_not(mask))

    if SAVE_DEBUG_IMAGES:
        save_image(Path(DEBUG_IMAGES_DIR) / 'mask.jpg', mask_not)

    if SAVE_DEBUG_IMAGES:
        img_add_green = img_add.copy()
        img_add_green[(mask_not == (255, 255, 255)).all(axis=2)] = (0, 135, 0)

        save_image(Path(DEBUG_IMAGES_DIR) / 'compose_green.jpg', img_add_green)

    mask_not_bin = cv2.cvtColor(mask_not, cv2.COLOR_BGR2GRAY)

    return img_add, mask_not_bin


def predict(imgs: list) -> None:
    '''predict first image without foreground objects from other images'''
    N = len(imgs)

    assert N > 1, f'Not Enough Images: {N}'

    if N > 5:
        log_warn(f'many images: {N}. It may take while')

    log_debug('detecting keypoints and computing descriptors...')

    kp, des = detect_keypoints_and_compute_descriptors(imgs)

    log_debug('finding all homographies...')

    H = find_all_homographies(kp, des)

    if not SKIP_HOMOGRAPHY_CHECK:

        log_debug('removing invalid homographies...')

        remove_invalid_homographies(H, imgs[0].shape)

    add_inverse_homographies(H)

    log_debug('aligning images...')

    aligned_images = align_images(imgs, H)

    N = len(aligned_images)

    assert N >= 2, f'Not Enough Aligned Images: {N}'

    log_debug(f'aligned {N-1} images')

    if SAVE_DEBUG_IMAGES:
        save_debug_images(aligned_images, 'aligned', 'png')
        log_debug('aligned images saved')

    log_debug('making masks...')

    masks = make_masks(aligned_images)

    if SAVE_DEBUG_IMAGES:
        save_debug_images(masks, 'mask', 'jpg')
        log_debug('mask images saved')

    log_debug('composing...')

    input_image, mask_image = compose(aligned_images, masks)

    save_image(INPUT_IMAGE_PATH, input_image)
    save_image(MASK_IMAGE_PATH, mask_image)

    if not NO_INPAINTING:
        log_debug('inpainting...')

        inpaint(INPUT_IMAGE_PATH, MASK_IMAGE_PATH, gpu=USE_GPU)

    log_debug('prediction finished')


def parse_args():
    parser = ArgumentParser(
        description='predict a base image without foreground objects from other images')

    parser.add_argument('BASE_IMAGE', help='base image file')
    parser.add_argument('IMAGES', nargs='+', help='other image files')
    parser.add_argument('-o', '--output-image', required=True,
                        help='output image')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose debug log output')
    parser.add_argument('-vv', '--more-verbose',    action='store_true',
                        help='more verbose progress bar output')
    parser.add_argument('--save-debug-images',    action='store_true',
                        help=f'save debug images to {DEBUG_IMAGES_DIR}')
    parser.add_argument('--skip-homography-check', action='store_true',
                        help='skip homography check')
    parser.add_argument('-i', '--inverse-mask', action='store_true',
                        help='inverse mask')
    parser.add_argument('--no-inpainting', action='store_true',
                        help='no inpainting')
    parser.add_argument('-gpu', '--use-gpu',  action='store_true',
                        help='use GPU for inpaint')

    args = parser.parse_args()

    return args


def main() -> None:
    global VERBOSE, MORE_VERBOSE, SAVE_DEBUG_IMAGES, SKIP_HOMOGRAPHY_CHECK, INVERSE_MASK, NO_INPAINTING, USE_GPU

    args = parse_args()

    base_image_file = args.BASE_IMAGE
    image_files = args.IMAGES
    output_image = args.output_image

    if args.verbose:
        VERBOSE = True
    if args.more_verbose:
        VERBOSE = True
        MORE_VERBOSE = True
    SAVE_DEBUG_IMAGES = args.save_debug_images
    SKIP_HOMOGRAPHY_CHECK = args.skip_homography_check
    INVERSE_MASK = args.inverse_mask
    NO_INPAINTING = args.no_inpainting
    USE_GPU = args.use_gpu

    imgs = [load_image(base_image_file)] + \
        [load_image(image_file) for image_file in image_files]

    predict(imgs)

    shutil.copy('out.png', output_image)


if __name__ == '__main__':
    main()
