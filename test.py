import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import cv2
import torch

class Test():
    def __init__(self):
        super().__init__()



    def getParser(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", type = str, required=True,
        help = "path to pytorch model")
        ap.add_argument("-o", "--output", type = str, required=True,
        help = "path to saved results")
        args = vars(ap.parse_args())
        return args

if __name__ == '__main__':
    
    args = Test().getParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = args["output"]
    print("[INFO] loading the KMNIST test dataset...")
    testData = KMNIST(root="data", train=False, download= True,
    transform = ToTensor())
    idxs = np.random.choice(range(0, len(testData)), size=(10,))
    testData = Subset(testData, idxs)

    testDataLoader = DataLoader(testData, batch_size = 1)

    model = torch.load(args["model"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for (image, label) in testDataLoader:
            origImage = image.numpy().squeeze(axis=(0,1))
            gtLabel = testData.dataset.classes[label.numpy()[0]]

            image = image.to(device)
            pred = model(image)

            idx = pred.argmax(axis=1).cpu().numpy()[0]
            predLabel = testData.dataset.classes[idx]

            origImage = np.dstack([origImage] * 3)
            origImage = imutils.resize(origImage, width=128)
            # draw the predicted class label on it
            color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
            image = cv2.putText(origImage, gtLabel, (2, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
            # display the result in terminal and show the input image
            print("[INFO] ground truth label: {}, predicted label: {}".format(
                gtLabel, predLabel))
            cv2.imwrite(filename+str(label)+".jpg", image)
            