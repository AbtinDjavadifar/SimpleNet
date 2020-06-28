import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
from PIL import Image

if __name__ == "__main__":
    data_dir = './Simple_Net/Images/chunks/'
    lbls_dir='./Simple_Net/Images/labels/'
    #Probabilities = open('Probabilities.txt', 'w+')

    test_transforms = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SimpleNet()
    #model.load_state_dict(torch.load("/home/aeroclub/Abtin/CASE/Simple_Net/WDmodel_Try1_212of200.pth"))
    model=torch.load('/home/aeroclub/Abtin/CASE/Simple_Net/Weights/WDmodel_Try1_499.pth')
    model.eval()

    for filename in [f for f in os.listdir(data_dir) if f.endswith(".jpg")]:
        filepath=os.path.join(data_dir, filename)
        image = Image.open(filepath)
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        _, prediction_test = torch.max(output.data, 1)
        Probabilities=open(lbls_dir + filename[:-4] + ".txt", "w+")
        Probabilities.write(str(torch.nn.Softmax()(output.data)) + "Class-" + str(prediction_test) + "\n")