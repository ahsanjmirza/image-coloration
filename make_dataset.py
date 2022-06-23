from argparse import ArgumentParser 
import numpy as np
from imageio import imread
import os
from tqdm import tqdm
from skimage import color

def build_dataset(args):
    x_patches = []
    y_patches = []
    patch_size = args.patchSize
    batch_ready_flag = 0
    batch_num = 0
    image_list = os.listdir(args.img)
    for f in tqdm(image_list, ncols=100):
        rgb = imread(os.path.join(args.img, f))
        (img_h, img_w) = (rgb.shape[0], rgb.shape[1])
        cond_size = img_w > patch_size and img_h > patch_size
        cond_chs = len(rgb.shape) == 3
        if cond_size and cond_chs:
            img = color.rgb2ycbcr(rgb)
            img = np.clip(img, 0, 255)
            grey_img = img[:, :, 0].reshape((img_h, img_w, 1))
            color_img = img[:, :, 1:]
            j = 0
            while j < img_w:
                if j > img_w - patch_size:
                    j = img_w - patch_size
                i = 0
                while i < img_h:
                    if i > img_h - patch_size:
                        i = img_h - patch_size
                    x_patches.append(grey_img[i:i+patch_size, j:j+patch_size, :])
                    y_patches.append(color_img[i:i+patch_size, j:j+patch_size, :])
                    batch_ready_flag += 1
                    if batch_ready_flag == args.batchSize:
                        x_patches = np.array(x_patches)
                        y_patches = np.array(y_patches)
                        np.save(os.path.join(args.dataset_path, 'grey', 'batch_'+str(batch_num)+'.npy'), np.uint8(x_patches))
                        np.save(os.path.join(args.dataset_path, 'color', 'batch_'+str(batch_num)+'.npy'), np.uint8(y_patches))
                        del x_patches, y_patches
                        x_patches = []
                        y_patches = []
                        batch_num += 1
                        batch_ready_flag = 0
                    i += patch_size
                j += patch_size

    return



if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess Dataset')

    parser.add_argument("--img", "-i",
                        required=True, 
                        type=str, 
                        help='Enter the name of the directory which contains all the images.')

    parser.add_argument("--dataset_path", "-d",
                        required=True, 
                        type=str, 
                        help='Enter the name of the train data directory.')

    parser.add_argument("--patchSize", "-p", 
                        required=False,
                        type=int,
                        help="Specify patch size to break images into", 
                        default=256)

    parser.add_argument("--batchSize", "-b",
                        required=False, 
                        type=int,
                        default=8,
                        help='Specify the batch size.')

    args = parser.parse_args()
    build_dataset(args)