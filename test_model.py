import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from vis_utils import visualize_grid

class ClassificationCNN(nn.Module):
    def __init__(self, input_dim=(3, 500, 500), num_filters=16, kernel_size=5,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=200,
                 num_classes=4, dropout=0.4):

        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        self.dropout = dropout
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim[0], num_filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.fc = nn.Linear(500000, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, self.dropout, True)
        out = F.relu(self.fc(out))
        out = self.fc2(out)
        return out

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)




def image_loader(image_name):
    """load image, returns cuda tensor"""
    imsize = 500
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).float()

    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel
        @allkernels: visualization all tensores
    '''

    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))




def main():
    model = torch.load('/home/dwinter/Dokumente/opencv_nn/models/classification_cnn.model')
    classes = ["Dominik", "Maren", "Nathaniel", "Alex"]
    for i in range(0,150):
        folder = classes[3]
        picture_pre = "alex"
        path = "/home/dwinter/Dokumente/opencv_nn/test_data/"+ folder + "/" + picture_pre + "" + str(i) + ".png"
        print(path)
        input = image_loader(path)
        output = model(input)
        _, pred = torch.max(output, 1)
        if pred.data.cpu().numpy()[0] == 1:
            print("Domi")
        elif pred.data.cpu().numpy()[0] == 2:
            print("Maren")
        elif pred.data.cpu().numpy()[0] == 3:
            print("Nath")
        elif pred.data.cpu().numpy()[0] == 0:
            print("Alex")

    # first (next) parameter should be convolutional
    '''
    conv_params = next(model.parameters()).detach().cpu().numpy()
    grid = visualize_grid(conv_params.transpose(0, 2, 3, 1))
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(50,50)
    plt.show()
    '''

if __name__ == "__main__":
    main()
