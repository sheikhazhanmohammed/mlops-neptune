from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms

#loading best model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(512,2)
model.load_state_dict(torch.load("./classificationModel.pt"))
model.eval()
target_layers = [model.layer4[1]]

inputImage = Image.open("./data/cats/c_1.jpg")
preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
inputTensor = preprocess(inputImage)
inputImage = np.array(inputImage)
inputImage = inputImage/255.0
inputTensor = inputTensor.unsqueeze(0)
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
targets = [ClassifierOutputTarget(0)]
grayscalecam = cam(input_tensor=inputTensor, targets=targets)
grayscalecam = grayscalecam[0, :]
visualization = show_cam_on_image(inputImage, grayscalecam, use_rgb=True)
plt.imshow(visualization)
plt.show()