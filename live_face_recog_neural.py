# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
        help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

model = torch.load('./classification_cnn.model')
input_test1 = image_loader(".test_data/Dominik/domi_12.png")
input_test2 = image_loader("./test_data/Nathaniel/nath_12.png")
input_test3 = image_loader("./test_data/Maren/maren_12.png")
input_test4 = image_loader("./test_data/Alex/alex12.png")

output_test = model(input_test1)
_, pred_test = torch.max(output_test, 1)
print("Domi is number:", pred_test.data.cpu().numpy()[0])
output_test = model(input_test2)
_, pred_test = torch.max(output_test, 1)
print("Nath is number:", pred_test.data.cpu().numpy()[0])
output_test = model(input_test3)
_, pred_test = torch.max(output_test, 1)
print("Maren is number:", pred_test.data.cpu().numpy()[0])
output_test = model(input_test4)
_, pred_test = torch.max(output_test, 1)
print("Alex is number:", pred_test.data.cpu().numpy()[0])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

width = 1280
height = 720
# loop over the frames from the video stream
counter = 0
while True:
    # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

    # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < args["confidence"]:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                try:
                    # neural net
                    head = frame[startY-40:endY+30, startX-40:endX+40]
                    cv2.imshow("Head", head)
                    neural_head = cv2.resize(head,(500, 500))
                    neural_head = Image.fromarray(neural_head)
                    loader = transforms.Compose([transforms.Scale(500), transforms.ToTensor()])
                    input = loader(neural_head).float()
                    #test = np.transpose(neural_head, (1,2,0))
                    #cv2.imwrite("test.png", test)
                except Exception as err:
                    print(err)
                    continue
                '''
                print(head.shape)
                channels = head[2]

                head_width = head[0]
                print(channels.shape)
                head_height = [1]
                print(head_height.shape)
                neural_head = np.array([channels, head_width, head_height], dtype=np.uint8)
                '''
                #input = image_loader("test.png")
                #input = torch.tensor(neural_head).cuda()
                input = Variable(input, requires_grad=True)
                input = input.unsqueeze(0).cuda()
                output = model(input.float())
                _, pred = torch.max(output, 1)
                if pred[0] == 1:
                    text = "Domi"
                elif pred[0] == 2:
                    text = "Maren"
                elif pred[0] == 3:
                    text = "Nath"
                elif pred[0] == 0:
                    text = "Alex"

                '''
                #save img
                show_head = cv2.resize(head,(500,500))
                cv2.imshow("Head", show_head)
                path = "/home/dwinter/Dokumente/opencv/val_data/Nathaniel/"
                save_name = "nath_" + str(counter) + ".png"
                cv2.imwrite(os.path.join(path, save_name), head)
                counter += 1
                '''
                # draw the bounding box of the face along with the associated
                # probability
                #text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


        frame = cv2.resize(frame,(width,height))    # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        if counter >= 500:
            break



# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
