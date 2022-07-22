from argparse import ArgumentParser 
import numpy as np
from imageio import imread, imwrite
import os
from tqdm import tqdm
from skimage import transform

def main(args):
    patch_size = args.patchSize
    patch_num = 0
    image_list = os.listdir(args.in_path)
    for f in tqdm(image_list, ncols=100):
        in_img = imread(os.path.join(args.in_path, f))
        (img_h, img_w) = (in_img.shape[0], in_img.shape[1])
        if img_h >= 2*patch_size and img_w >= 2*patch_size:
            j = 0
            while j < img_w:
                if j > img_w - patch_size:
                    j = img_w - patch_size
                i = 0
                while i < img_h:
                    if i > img_h - patch_size:
                        i = img_h - patch_size
                    imwrite(os.path.join(args.out_path, str(patch_num)+'.jpg'), np.uint8(in_img[i:i+patch_size, j:j+patch_size]))
                    patch_num += 1
                    i += patch_size
                j += patch_size
        else:
            in_img = transform.resize(
                image=np.float32(in_img[:, :, :3]), 
                output_shape=(patch_size, patch_size),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
            in_img = np.uint8(np.clip(in_img, 0, 255))
            imwrite(os.path.join(args.out_path, str(patch_num)+'.jpg'), np.uint8(in_img))
            patch_num += 1
    return



if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess Dataset')

    parser.add_argument("--in_path", "-i",
                        required=True, 
                        type=str, 
                        help='Enter the path of the directory which contains all the input images.')
    
    parser.add_argument("--out_path", "-o",
                        required=True, 
                        type=str, 
                        help='Enter the path of the output directory.')

    parser.add_argument("--patchSize", "-p", 
                        required=False,
                        type=int,
                        help="Specify patch size to break images into", 
                        default=256)

    args = parser.parse_args()
    main(args)