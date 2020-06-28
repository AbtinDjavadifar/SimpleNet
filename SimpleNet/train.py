import torch
import torch.nn as nn
import torchvision.datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output

# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    global num_epochs
    lr = 0.001

    #    if epoch > round(num_epochs / 1.11):
    #        lr = lr / 1000000
    if epoch > round(num_epochs * 0.8):
        lr = lr / 100000
    elif epoch > round(num_epochs * 0.6):
        lr = lr / 10000
    elif epoch > round(num_epochs * 0.4):
        lr = lr / 1000
    elif epoch > round(num_epochs * 0.2):
        lr = lr / 100
    elif epoch > round(num_epochs * 0.1):
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_models(epoch):
    # torch.save(model.state_dict(), 'WDmodel_Try1_{}of200.pth'.format(epoch))
    torch.save(model, './Weights/WDmodel_Try1_{}.pth'.format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        prediction = prediction.cpu().numpy()
        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 200

    return test_acc

def train(num_epochs):
    best_acc = 0.0
    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc.item() / 1500
        train_loss = train_loss / 1500

        # Evaluate on the test set
        # test_acc = test()

        model.eval()
        test_acc = 0.0
        background_acc = 0.0
        fabric_acc = 0.0
        gripper_acc = 0.0
        wrinkle_acc = 0.0

        for i, (images_test, labels_test) in enumerate(test_loader):

            if cuda_avail:
                images_test = Variable(images_test.cuda())
                labels_test = Variable(labels_test.cuda())

            background_label = (labels_test == 0)
            fabric_label = (labels_test == 1)
            gripper_label = (labels_test == 2)
            wrinkle_label = (labels_test == 3)

            # Predict classes using images from the test set
            outputs_test = model(images_test)
            _, prediction_test = torch.max(outputs_test.data, 1)

            background_output = (prediction_test == 0)
            fabric_output = (prediction_test == 1)
            gripper_output = (prediction_test == 2)
            wrinkle_output = (prediction_test == 3)

            # prediction_test = prediction_test.cpu().numpy()
            test_acc += torch.sum(prediction_test == labels_test.data)
            background_acc += torch.sum(background_label == background_output)
            wrinkle_acc += torch.sum(wrinkle_label == wrinkle_output)
            fabric_acc += torch.sum(fabric_label == fabric_output)
            gripper_acc += torch.sum(gripper_label == gripper_output)

        # Compute the average acc and loss over all 10000 test images
        test_acc = test_acc.item() / 200
        background_acc = background_acc.item() / 200
        fabric_acc = fabric_acc.item() / 200
        gripper_acc = gripper_acc.item() / 200
        wrinkle_acc = wrinkle_acc.item() / 200

        # Save the model if the test acc is greater than our current best
        if test_acc > 0.88:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))
        print(
            "Background {}, Fabric {}, Gripper {}, Wrinkle {}".format(background_acc, fabric_acc, gripper_acc, wrinkle_acc))

if __name__ == "__main__":

    # Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
    train_transformations = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 32

    # Load the training set
    # train_set = CIFAR10(root="./data",train=True,transform=train_transformations,download=True)
    train_set = torchvision.datasets.ImageFolder(root='./Training_Images/Training_Set/',
                                                 transform=train_transformations)
    # Create a loder for the training set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define transformations for the test set
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    # Load the test set, note that train is set to False
    # test_set = CIFAR10(root="./data",train=False,transform=test_transformations,download=True)
    test_set = torchvision.datasets.ImageFolder(root="./Training_Images/Test_Set/", transform=train_transformations)
    # Create a loder for the test set, note that both shuffle is set to false for the test loader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Check if gpu support is available
    cuda_avail = torch.cuda.is_available()

    # Create model, optimizer and loss function
    model = SimpleNet(num_classes=4)

    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    train(10)