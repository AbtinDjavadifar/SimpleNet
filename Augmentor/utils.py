import Augmentor
from invoke import run
from PIL import Image
from random import randint
import glob
import os

def augmentor():

    WrinklePath="./Wrinkle_templates"

    p = Augmentor.Pipeline(WrinklePath)
    p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.skew_tilt(probability=1)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.skew_left_right(probability=1)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.skew_top_bottom(probability=1)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.skew_corner(probability=1)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.skew(probability=1)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.random_distortion(probability=1, grid_width=16, grid_height=16, magnitude=8)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.shear(probability=1, max_shear_left=20, max_shear_right=20)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.crop_random(probability=1, percentage_area=0.7)
    p.process()

    p = Augmentor.Pipeline(WrinklePath)
    p.flip_random(probability=1)
    p.process()

    #p.status
    #p.sample(100)

def augmentor_test():

    # =============================================================================
    # def Augment(inputpath, cmdspath):
    #     cmds = open(cmdspath).readlines()
    #     for i in range(len(cmds)):
    #         p = Augmentor.Pipeline(inputpath)
    #         run(cmds[i])
    #         p.process()
    #
    # inpath="/home/aeroclub/Abtin/Augmentor/figure_skating_templates"
    # cmpath="/home/aeroclub/Abtin/Augmentor/cmds.txt"
    # Augment(inpath, cmpath)
    # =============================================================================

    p = Augmentor.Pipeline("./Augmentor/figure_skating_templates")
    cmds = "p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)"
    run(cmds, hide=True, warn=True)
    p.process()

def copy():
    temp_dir = "./Wrinkle_templates/output"
    bckg_dir = "./Fabric_templates"
    dtst_dir = "./Generated_files/Images/"
    labls_dir = "./Generated_files/Labels/xywh/"
    num = 0

    for bckg_img in glob.glob(bckg_dir + "/*.JPG"):
        bckg = Image.open(bckg_img)
        bckg_w, bckg_h = bckg.size
        for tmp_img in glob.glob(temp_dir + "/*.JPG"):
            tmp = Image.open(tmp_img)
            tmp_w, tmp_h = tmp.size

            # print(bckg_w, bckg_h, tmp_w, tmp_h)

            for i in range(2):
                new = bckg.copy()
                if bckg_w - tmp_w > 1 and bckg_h - tmp_h > 1:
                    num += 1
                    x = randint(1, bckg_w - tmp_w)
                    y = randint(1, bckg_h - tmp_h)
                    offset = (x, y)
                    new.paste(tmp, offset)
                    # new.show()
                    new.save(dtst_dir + str(num).zfill(6) + ".jpg")
                    f = open(labls_dir + str(num).zfill(6) + ".txt", "w+")
                    f.write("1\n")
                    # 1 is the number of class
                    f.write(str(x) + " " + str(y) + " " + str(tmp_w) + " " + str(tmp_h))
                    f.close()

def train_test_list_creator():
    dataset_path = "./Generated_files/Images/wrinkle/"

    # Percentage of images to be used for the test set
    percentage_test = 10;

    # Create and/or truncate train.txt and test.txt
    file_train = open('train.txt', 'w+')
    file_test = open('test.txt', 'w+')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test + 1:
            counter = 1
            file_test.write(dataset_path + title + '.jpg' + "\n")
        else:
            file_train.write(dataset_path + title + '.jpg' + "\n")
            counter = counter + 1

def YOLO3_format_converter():

    classes = ["wrinkle"]

    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        print(x)
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        print(dw)

        print(dh)
        return (x, y, w, h)

    """ Configure Paths"""
    mypath = "./Generated_files/Labels/xywh/"
    outpath = "./Generated_files/Labels/YOLO/"

    cls = "wrinkle"
    if cls not in classes:
        exit(0)
    cls_id = classes.index(cls)

    wd = os.getcwd()
    list_file = open('%s/%s_list.txt' % (wd, cls), 'w')

    """ Get input text file list """
    txt_name_list = []
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        txt_name_list.extend(filenames)
        break
    print(txt_name_list)

    """ Process """
    for txt_name in txt_name_list:
        # txt_file =  open("Labels/stop_sign/001.txt", "r")

        """ Open input text files """
        txt_path = mypath + txt_name
        print("Input:" + txt_path)
        txt_file = open(txt_path, "r")
        lines = txt_file.read().split('\n')  # for ubuntu, use "\r\n" instead of "\n"

        """ Open output text files """
        txt_outpath = outpath + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "w")

        """ Convert the data to YOLO format """
        ct = 0
        for line in lines:
            # print('lenth of line is: ')
            # print(len(line))
            # print('\n')
            if (len(line) >= 2):
                ct = ct + 1
                print(line + "\n")
                elems = line.split(' ')
                print(elems)
                xmin = float(elems[0])
                xmax = float(elems[2]) + float(elems[0])
                ymin = float(elems[1])
                ymax = float(elems[3]) + float(elems[1])
                #
                img_path = str('%s/Generated_files/Images/%s/%s.jpg' % (wd, cls, os.path.splitext(txt_name)[0]))
                # t = magic.from_file(img_path)
                # wh= re.search('(\d+) x (\d+)', t).groups()
                im = Image.open(img_path)
                w = int(im.size[0])
                h = int(im.size[1])
                # w = int(xmax) - int(xmin)
                # h = int(ymax) - int(ymin)
                # print(xmin)
                print(w, h)
                b = (xmin, xmax, ymin, ymax)
                bb = convert((w, h), b)
                print(bb)
                txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        """ Save those images with bb into list"""
        if (ct != 0):
            list_file.write('%s/images/%s/%s.JPEG\n' % (wd, cls, os.path.splitext(txt_name)[0]))

    list_file.close()