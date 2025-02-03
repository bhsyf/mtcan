import random
import os


def main():
    val_ratio = 1 / 5.0  # 4:1 train:val
    dataset_dir = ""
    txt_dst = ""
    img_dir = os.path.join(dataset_dir, "A")
    train_list_txt = open(os.path.join(txt_dst, "train_list_Landsat-SCD.txt"), "w")
    val_list_txt = open(os.path.join(txt_dst, "val_list_Landsat-SCD.txt"), "w")

    img_list = os.listdir(img_dir)
    valid_list = []
    train_list = []
    val_list = []
    for it in img_list:
        it_ext = it[-4:]
        if it_ext == ".png":
            valid_list.append(it)
    num_pics = len(valid_list)
    num_val = int(num_pics * val_ratio)
    num_train = num_pics - num_val
    print(f"-all imgs: {num_pics}, {num_train} for train, {num_val} for val and test.")

    random.shuffle(valid_list)
    for idx, it in enumerate(valid_list):
        if idx < num_train:
            train_list.append(it)
        else:
            val_list.append(it)

    for it in train_list:
        train_list_txt.write(it + "\n")
    for it in val_list:
        val_list_txt.write(it + "\n")

    print("-split finished.")


if __name__ == "__main__":
    main()
