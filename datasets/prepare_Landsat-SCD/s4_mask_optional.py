import numpy as np
from skimage import io
import os


def main():
    """Remove blank areas(color white in imgAB, color black in rgb_label, RGB [0,0,0]).
    Only the color labels are masked!
    Careful: in index labels, blank areas are labeled as 0(same as nochange)!"""
    dataset_dir = ""
    ImgDirA = os.path.join(dataset_dir, "A")
    ImgDirB = os.path.join(dataset_dir, "B")
    LabelDirA_rgb = os.path.join(dataset_dir, "labelA_rgb_masked")
    LabelDirB_rgb = os.path.join(dataset_dir, "labelB_rgb_masked")

    data_list = os.listdir(ImgDirA)
    for idx, it in enumerate(data_list):
        if it[-4:] == ".png":
            Img_pathA = os.path.join(ImgDirA, it)
            Img_pathB = os.path.join(ImgDirB, it)
            label_pathA_rgb = os.path.join(LabelDirA_rgb, it)
            label_pathB_rgb = os.path.join(LabelDirB_rgb, it)

            imgA = io.imread(Img_pathA)
            imgB = io.imread(Img_pathB)
            labelA_rgb = io.imread(label_pathA_rgb)
            labelB_rgb = io.imread(label_pathB_rgb)

            # blank areas are white in imgAB
            invalid_maskA = np.all(imgA == 255, axis=2)
            invalid_maskB = np.all(imgB == 255, axis=2)
            invalid_mask = invalid_maskA * invalid_maskB
            valid_mask = np.logical_not(invalid_mask)
            valid_mask = np.expand_dims(valid_mask, 2).repeat(3, axis=2)
            labelA_rgb = labelA_rgb * valid_mask
            labelB_rgb = labelB_rgb * valid_mask

            io.imsave(label_pathA_rgb, labelA_rgb, check_contrast=False)
            io.imsave(label_pathB_rgb, labelB_rgb, check_contrast=False)

    print("-finished masking blank areas.")


if __name__ == "__main__":
    main()
