"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import RandomSampler
import torchvision.models as models
from matplotlib import pyplot as plt


class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 1e-5}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):

        optim = self.optim(model.parameters(), **self.optim_args)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        self._reset_histories()

        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        for epoch in range(0,num_epochs):
            #val_iter = iter(val_loader)
            train_loss = 0.0

            train_scores = []
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()


                outputs = model(inputs)
                outputs = outputs.cuda()

                loss = criterion(outputs, labels)


                optim.zero_grad()

                loss.backward()

                optim.step()

                train_loss += loss.item()
                self.train_loss_history.append(loss.item())
                _, argmax = torch.max(outputs, 1)
                train_accuracy = (labels == argmax.squeeze()).float().mean()
                self.train_acc_history.append(train_accuracy)

                _, argmax = torch.max(outputs, 1)
                train_scores.extend((argmax == labels).data.cpu().numpy())

                if i % 100 == 99 :
                    train_accuracy = np.mean(train_scores)
                    print('[ Iteration %1d/%1d] TRAIN loss: %.3f TRAIN accuracy %.3f' %
                     (i + 1,iter_per_epoch, train_loss / 100.0, train_accuracy))

                    train_loss = 0.0

            scores = []
            for T_inputs, T_targets in val_loader:
                T_inputs, T_targets = T_inputs.to(device), T_targets.to(device)

                T_outputs = model(T_inputs)
                _, preds = torch.max(T_outputs, 1)
                scores.extend((preds == T_targets).data.cpu().numpy())

            print('Val set accuracy: %f' % np.mean(scores))
            self.val_acc_history.append(np.mean(scores))
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')


class ClassificationCNN(nn.Module):
    def __init__(self, input_dim=(3, 500, 500), num_filters=16, kernel_size=5,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=120,
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

        '''
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(input_dim[0], num_filters, kernel_size=kernel_size, stride=1, padding=input_dim[0]).cuda()
        self.conv1_weights = self.conv1.weight.data
        self.conv1_weights *= weight_scale
        self.pool = nn.MaxPool2d(pool, stride_pool, padding=0)


        self.fc1 = nn.Linear(self.num_filters * int(self.input_dim[1]/pool) * int(self.input_dim[1]/pool), self.hidden_dim, True).cuda()
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes, True).cuda()
        '''


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, self.dropout, True)
        out = F.relu(self.fc(out))
        out = self.fc2(out)
        return out
        '''
        x = F.relu(self.conv1(x))

        x = self.pool(x)

        max_pool_dim = x.shape[2]

        x = x.view(-1, self.num_filters * max_pool_dim * max_pool_dim)

        x = F.dropout(x, self.dropout, True)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
        '''
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = ImageFolder(root='train_data', transform=ToTensor())
    print("Train data specs:")
    print(train_data.classes)
    print(len(train_data))
    print("\n")



    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=10)



    val_data = ImageFolder(root='val_data', transform=ToTensor())
    print("Val data specs:")

    print(val_data.classes)
    print(len(val_data))
    print("\n")



    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=10)




    model = ClassificationCNN()
    model.to(device)
    solver = Solver(optim_args={"lr": 1e-2})
    solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=2)



    test_data = ImageFolder(root='test_data', transform=ToTensor())
    print("Test data specs:")
    print(test_data.classes)
    print(len(test_data))
    print("\n")

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=10)





    scores = []
    model.eval()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)


        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        scores.extend((preds == targets).data.cpu().numpy())

    print('Test set accuracy: %f' % np.mean(scores))

    model.save("models/classification_cnn.model")

if __name__ == "__main__":
    main()
