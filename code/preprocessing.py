import os
import time
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree


download_image_dir = '../Images_png'
selected_image_dir = '../selected_images'
processed_image_dir = '../processed_images'
smoothed_image_dir = '../smoothed_images'


def clear_image_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            rmtree(file_path)


def copy_images():
    sub_dirs = [subDir[0].split('/')[-1] for subDir in os.walk(download_image_dir)]

    info = pd.read_csv('../info/info.csv')

    count = 0

    for index, row in info.iterrows():
        folder = '_'.join(row['File_name'].split('_')[:-1])

        if folder in sub_dirs:
            src = download_image_dir + '/' + folder + '/' + row['File_name'].split('_')[-1]
            dest = selected_image_dir + '/' + folder + '_' + row['File_name'].split('_')[-1]
            copyfile(src, dest)

            count = count + 1

    print(count, 'images copied')


def convert_images():
    info = pd.read_csv('../info/info.csv')

    image_to_window = {}

    for index, row in info.iterrows():
        image_to_window[row['File_name']] = row['DICOM_windows']

    for filename in os.listdir(selected_image_dir):
        selected_file_path = os.path.join(selected_image_dir, filename)
        processed_file_path = os.path.join(processed_image_dir, filename)
        smoothed_file_path = os.path.join(smoothed_image_dir, filename)

        # images read in as 16 bit
        img = cv2.imread(selected_file_path, cv2.IMREAD_UNCHANGED)

        # convert to original Hounsfield units
        img = img.astype(np.float32, copy=False) - 32768

        # scale based on intensity windows
        min_max = image_to_window[filename].split(', ')
        window_min = float(min_max[0])
        window_max = float(min_max[1])

        for x in range(0, 512):
            for y in range(0, 512):
                img[x, y] = min(255, max(0, (img[x, y] - window_min) / (window_max - window_min) * 255))

        cv2.imwrite(processed_file_path, img)

        smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)

        cv2.imwrite(smoothed_file_path, smoothed_img)


if __name__ == "__main__":
    start_time = time.time()

    copy_images()
    convert_images()

    end_time = time.time()
    print("Execution Time: %s seconds" % (end_time - start_time))
