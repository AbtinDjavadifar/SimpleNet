import shutil
import random
from PIL import Image
import os

def devide_to_train_test(dir = "./Simple_Net", k = 4):

    Wrinkle_dirpath_source = os.path.join(dir, "DLR_Dataset/Wrinkle/")
    Fabric_dirpath_source = os.path.join(dir, "DLR_Dataset/Fabric/")
    Gripper_dirpath_source = os.path.join(dir, "DLR_Dataset/Gripper/")
    Background_dirpath_source = os.path.join(dir, "DLR_Dataset/Background/")

    Wrinkle_dirpath_dest_train = os.path.join(dir, "Training_Images/Training_Set/Wrinkle/")
    Fabric_dirpath_dest_train = os.path.join(dir, "Training_Images/Training_Set/Fabric/")
    Gripper_dirpath_dest_train = os.path.join(dir, "Training_Images/Training_Set/Gripper/")
    Background_dirpath_dest_train = os.path.join(dir, "Training_Images/Training_Set/Background/")

    Wrinkle_dirpath_dest_test = os.path.join(dir, "Training_Images/Test_Set/Wrinkle/")
    Fabric_dirpath_dest_test = os.path.join(dir, "Training_Images/Test_Set/Fabric/")
    Gripper_dirpath_dest_test = os.path.join(dir, "Training_Images/Test_Set/Gripper/")
    Background_dirpath_dest_test = os.path.join(dir, "Training_Images/Test_Set/Background/")

    classes = {}
    classes["Wrinkle"] = {"src": Wrinkle_dirpath_source, "train": Wrinkle_dirpath_dest_train,
                          "test": Wrinkle_dirpath_dest_test, "num": 250*k}
    classes["Fabric"] = {"src": Fabric_dirpath_source, "train": Fabric_dirpath_dest_train,
                          "test": Fabric_dirpath_dest_test, "num": 150 * k}
    classes["Gripper"] = {"src": Gripper_dirpath_source, "train": Gripper_dirpath_dest_train,
                          "test": Gripper_dirpath_dest_test, "num": 150 * k}
    classes["Background"] = {"src": Background_dirpath_source, "train": Background_dirpath_dest_train,
                          "test": Background_dirpath_dest_test, "num": 200 * k}

    def random_picker(target):

        filenames = random.sample(os.listdir(classes[target]["src"]), classes[target]["num"] + 50)

        for fname in filenames[0:classes[target]["num"]]:
            srcpath = os.path.join(classes[target]["src"], fname)
            shutil.copy(srcpath, classes[target]["train"])

        for fname in filenames[classes[target]["num"]:]:
            srcpath = os.path.join(classes[target]["src"], fname)
            shutil.copy(srcpath, classes[target]["test"])

    for target in classes.keys():
        random_picker(target)

def slicer(height=81, width=108, images_path = "./Simple_Net/Images/def_ply011", chunks_path = "./Simple_Net/Images/chunks/"):
    for file_name in [f for f in os.listdir(images_path) if f.endswith(".jpg")]:
        file_path = os.path.join(images_path, file_name)
        im = Image.open(file_path)
        img_width, img_height = im.size
        for i in range(0, img_width, width):
            for j in range(0, img_height, height):
                box = (i, j, i + width, j + height)
                a = im.crop(box)
                a.save(os.path.join(chunks_path, file_name[:-4] + "-X-%s-Y-%s" % (i, j) + ".jpg"))

def demspter_shafer_example():

    from pyds import MassFunction

    v1 = MassFunction({'w': 0.6, 'f': 0.3, 'g': 0.1, 'b': 0.0})
    v2 = MassFunction({'w': 0.5, 'f': 0.4, 'g': 0.1, 'b': 0.0})
    v3 = MassFunction({'w': 0.7, 'f': 0.2, 'g': 0.1, 'b': 0.0})
    v4 = MassFunction({'w': 0.3, 'f': 0.3, 'g': 0.4, 'b': 0.0})

    print('Dempster\'s combination rule for m_1 and m_2 =', v1 & v2 & v3 & v4)