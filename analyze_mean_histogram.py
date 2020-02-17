import os
import numpy as np
import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import argparse

from data import pointcloud_loader_kitti, removePoints, makeBVFeature
from lyft2kitti_converter import Lyft2KittiConverter
from skimage.measure import compare_ssim

LYFT_LIDAR_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/velodyne'
VALIDATION_FILE_LIST_PATH = '/home/user/work/master_thesis/code/Complex-YOLOv3/data/LYFT/ImageSets/valid.txt'
BEV_BOUNDARY = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}
IMG_HEIGHT = 480
IMG_WIDTH = 480
UNIT_CONFIG = "/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/config.yaml"
UNIT_CHECKPOINT = "/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/checkpoints/gen_00010000.pt"
lyft2kitti_conv = Lyft2KittiConverter(UNIT_CONFIG, UNIT_CHECKPOINT, a2b=1, seed=5)


def read_txt_file(path):
    with open(path) as f:
        return f.readlines()


def get_file_names(file_list_path):
    content = read_txt_file(file_list_path)
    filename_list = [x.strip() for x in content if x != '']
    return filename_list


def print_min_max(np_array):
    print("min: ", np.amin(np_array))
    print("max: ", np.amax(np_array))


def load_bevs(filename):
    input_file_path = os.path.join(LYFT_LIDAR_PATH, filename+'.bin')
    # original bev
    pointcloud_original = pointcloud_loader_kitti(input_file_path)
    pointcloud_original_filtered = removePoints(pointcloud_original, BEV_BOUNDARY)  # filter point cloud points inside fov
    discretization = (BEV_BOUNDARY["maxX"] - BEV_BOUNDARY["minX"]) / IMG_HEIGHT
    bev_original = makeBVFeature(pointcloud_original_filtered, BEV_BOUNDARY, IMG_HEIGHT, IMG_WIDTH,
                              discretization)  # create Bird's Eye View
    bev_original_2channel = bev_original[:, :, :2]
    # transformed bev
    bev_transformed_raw = lyft2kitti_conv.transform(bev_original_2channel)
    bev_transformed = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    bev_transformed[:, :, 0] = bev_transformed_raw[0, :, :]
    bev_transformed[:, :, 1] = bev_transformed_raw[1, :, :]

    return bev_original, bev_transformed


def calc_difference_img(image_a, image_b):
    (score, diff) = compare_ssim(image_a, image_b, full=True)
    diff_img = (diff * 255).astype("uint8")
    return diff_img


def create_histogram(image, num_bin):
    return cv2.calcHist([image], [0], None, [num_bin], [0, 255])


def plot_histogram(histogram, label, title, show=False):
    plt.plot(histogram, label=label)
    plt.xlim([0, histogram.shape[0]])
    plt.yscale('log')
    plt.xlabel('Grayscale value')
    plt.ylabel('Number of pixels')
    plt.title(title)
    plt.legend(loc="upper right")
    if show:
        plt.show()


def save_curr_histogram(output_filename):
    plt.savefig(output_filename, dpi=300)


def create_height_density_images(filename, pointcloud_original, pointcloud_transformed):
    # density and height for original pointcloud
    pointcloud_original_filename = filename + '_original'
    output_filename_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_height.png' % pointcloud_original_filename)
    output_filename_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_density.png' % pointcloud_original_filename)
    imageio.imwrite(output_filename_density, pointcloud_original[:, :, 0])
    imageio.imwrite(output_filename_height, pointcloud_original[:, :, 1])

    # density and height for transformed pointcloud
    pointcloud_transformed_filename = filename + '_transformed'
    output_filename_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_height.png' % pointcloud_transformed_filename)
    output_filename_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_density.png' % pointcloud_transformed_filename)
    imageio.imwrite(output_filename_density, pointcloud_transformed[:, :, 0])
    imageio.imwrite(output_filename_height, pointcloud_transformed[:, :, 1])

    # show differences as images
    diff_img_density = calc_difference_img(pointcloud_original[:, :, 0], pointcloud_transformed[:, :, 0])
    diff_img_height = calc_difference_img(pointcloud_original[:, :, 1], pointcloud_transformed[:, :, 1])
    output_filename_diff_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_diff_density.png' % filename)
    output_filename_diff_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_diff_height.png' % filename)
    imageio.imwrite(output_filename_diff_density, diff_img_density[:, :, 0])
    imageio.imwrite(output_filename_diff_height, diff_img_height[:, :, 1])


def subplot_images(pointcloud_original, pointcloud_transformed):
    f, axarr = plt.subplots(2, 2)
    f.set_figheight(15)
    f.set_figwidth(15)
    axarr[0, 0].imshow(pointcloud_original[:, :, 0])
    axarr[0, 1].imshow(pointcloud_original[:, :, 1])
    axarr[1, 0].imshow(pointcloud_transformed[:, :, 0])
    axarr[1, 1].imshow(pointcloud_transformed[:, :, 1])
    axarr[0, 0].axis('off')
    axarr[0, 1].axis('off')
    axarr[1, 0].axis('off')
    axarr[1, 1].axis('off')
    plt.show()


def main():
    print("UNIT CONFIG: ", UNIT_CONFIG)
    print("UNIT CHECKPOINT: ", UNIT_CHECKPOINT)
    filename_list = get_file_names(VALIDATION_FILE_LIST_PATH)
    hist_original_sum = None
    hist_transformed_sum = None
    number_of_files = 0
    for filename in filename_list:
        print("Processing: ", filename)
        bev_original, bev_transformed = load_bevs(filename)

        # convert from float32 to int8
        bev_original_int = (np.round_(bev_original * 255)).astype(np.uint8)
        bev_transformed_int = (np.round_(bev_transformed * 255)).astype(np.uint8)

        # create histograms
        hist_original = create_histogram(bev_original_int[:, :, 1], 255)
        hist_transformed = create_histogram(bev_transformed_int[:, :, 1], 255)

        # add to sum histograms
        if hist_original_sum is not None:
            hist_original_sum += hist_original
        else:
            hist_original_sum = hist_original
        if hist_transformed_sum is not None:
            hist_transformed_sum += hist_transformed
        else:
            hist_transformed_sum = hist_transformed

        # count number of files
        number_of_files += 1

    # calculate mean histogram
    hist_original_mean = np.true_divide(hist_original_sum, number_of_files)
    hist_transformed_mean = np.true_divide(hist_transformed_sum, number_of_files)
    plot_histogram(hist_original_mean, label='lyft', title='BEV Height Histogram')
    plot_histogram(hist_transformed_mean, label='lyft2kitti', title='BEV Height Histogram')
    save_curr_histogram(output_filename='mean_height_histogram')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opts = parser.parse_args()

    UNIT_CONFIG = opts.config
    UNIT_CHECKPOINT = opts.checkpoint
    main()
