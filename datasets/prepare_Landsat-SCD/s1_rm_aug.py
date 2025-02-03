import shutil
import os

AUG_NAMES = ["rotate", "Crop", "ZheDang", "Zhedang"]


def is_aug(it_name):
    for aug_name in AUG_NAMES:
        if aug_name in it_name:
            return True
    return False


def main():
    """remove augmented imgs"""
    dataset_src = ""
    dataset_dst = ""  # dst
    img1_src = os.path.join(dataset_src, "A")
    img1_dst = os.path.join(dataset_dst, "A")
    img2_src = os.path.join(dataset_src, "B")
    img2_dst = os.path.join(dataset_dst, "B")
    label_src = os.path.join(dataset_src, "label")
    label_dst = os.path.join(dataset_dst, "label")

    if not os.path.exists(img1_dst):
        os.makedirs(img1_dst)
    if not os.path.exists(img2_dst):
        os.makedirs(img2_dst)
    if not os.path.exists(label_dst):
        os.makedirs(label_dst)

    data_list = os.listdir(label_src)
    nums = 0
    for img_name in data_list:
        if not (img_name[-4:] == ".png" and is_aug(img_name)):
            label_src_path = os.path.join(label_src, img_name)
            label_dst_path = os.path.join(label_dst, img_name)
            shutil.copyfile(label_src_path, label_dst_path)
            img1_src_path = os.path.join(img1_src, img_name)
            img1_dst_path = os.path.join(img1_dst, img_name)
            shutil.copyfile(img1_src_path, img1_dst_path)
            img2_src_path = os.path.join(img2_src, img_name)
            img2_dst_path = os.path.join(img2_dst, img_name)
            shutil.copyfile(img2_src_path, img2_dst_path)
            nums += 1
            print(img_name)
    print(f"-finished removing auged imgs, now we have {nums} imgs.")  # 2385 imgs


if __name__ == "__main__":
    main()
