import os

if __name__ == "__main__":

    ground_truth_path = "./Simple_Net/Images/groundtruth"
    labels_path = "./Simple_Net/Images/labels"
    list_l = os.listdir(labels_path)
    list_gt = os.listdir(ground_truth_path)
    count = 0
    train_acc = 0
    l = 0

    for filenamel in [f for f in list_l if f.endswith(".txt")]:
        filepath = os.path.join(labels_path, filenamel)
        file = open(filepath, "r")
        sent = file.read()
        gt = int(sent[-21])

        for filenamegt in [f for f in list_gt if f.endswith(".txt")]:
            if filenamel == filenamegt:
                count += 1
                filepath = os.path.join(ground_truth_path, filenamel)
                file = open(filepath, "r")
                sent = file.read()[:-1]
                print(count)
                if sent == 'Wrinkle':
                    l = 3
                elif sent == 'Background':
                    l = 0
                elif sent == 'Fabric':
                    l = 1
                elif sent == 'Gripper':
                    l = 2

                if gt == l:
                    train_acc += 1

    train_acc = train_acc / count
    print("number of images: %d", count)
    print("test accuracy: %d %", train_acc * 100)