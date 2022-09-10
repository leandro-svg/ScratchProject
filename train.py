import matplotlib
matplotlib.use("Agg")

from model.ConvNet import ConvNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as numpy
import numpy as np
import argparse
import torch
import time



class Trainer():
    def __init__(self, args):
        super().__init__()
        print("Training begins...")

    def DataLoading(self):
        print("[INFO] loading the KMINST data...")
        self.trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
        self.testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

        print("[INFO] generating the train/validatiion split...")

        numTrainSamples = int(len(self.trainData)*args.TRAIN_SPLIT)
        numValSamples = int(len(self.trainData)*args.VAL_SPLIT)
        (self.trainData, valData) = random_split(self.trainData, 
                                [numTrainSamples, numValSamples],
                                generator = torch.Generator().manual_seed(42))

        self.trainDataLoader = DataLoader(self.trainData, shuffle=True, batch_size = args.BATCH_SIZE)
        self.valDataLoader = DataLoader(valData, shuffle=False, batch_size = args.BATCH_SIZE)
        self.testDataLoader = DataLoader(self.testData, shuffle=False, batch_size = args.BATCH_SIZE)


        self.trainSteps = len(self.trainDataLoader.dataset) // args.BATCH_SIZE
        self.valSteps = len(self.valDataLoader.dataset) // args.BATCH_SIZE
        
        self.H = {
            "train_loss" : [],
            "train_acc" : [],
            "val_loss" : [],
            "val_acc" : []
        }
        

    def modelLoader(self):
        print("[INFO] Initializing the ConvNet model")

        self.model = ConvNet(num_channels=1, classes=len(self.trainData.dataset.classes)).to(device)

        self.opt = Adam(self.model.parameters(), lr=args.init_lr)
        self.lossFn = nn.NLLLoss()

    def training(self):

        print("[INFO] training our neural network...")
        startTime = time.time()

        for e in range(0, args.EPOCH):
            self.model.train()

            totalTrainLoss = 0
            totalValLoss = 0

            trainCorrect = 0
            valCorrect = 0

            for (x,y) in self.trainDataLoader:
                (x,y) = (x.to(device), y.to(device))

                pred =  self.model(x)
                loss = self.lossFn(pred, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
            with torch.no_grad():
                self.model.eval()

                for (x,y) in self.valDataLoader:
                    (x,y) = (x.to(device), y.to(device))

                    pred = self.model(x)
                    totalValLoss += self.lossFn(pred, y)

                    valCorrect += (pred.argmax(1) == y).type(
                        torch.float).sum().item()

            avgTrainLoss = totalTrainLoss / self.trainSteps
            avgValLoss = totalValLoss / self.valSteps

            trainCorrect = trainCorrect/ len(self.trainDataLoader.dataset)
            valCorrect = valCorrect  /len(self.valDataLoader.dataset)

            self.H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            self.H["train_acc"].append(trainCorrect)
            self.H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            self.H["val_loss"].append(valCorrect)

            print("[INFO] EPOCH: {}/{}".format(e + 1, args.EPOCH))
            print("Train loss : {:.6f}, Train accuracay : {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print("Val loss : {:.6f}, Val accuracay : {:.4f}".format(
                avgValLoss, valCorrect))
            

        endTime = time.time()
        print("[INFO] total time taken to trian the model : {:.2f}s".format(
        endTime - startTime))

    def eval(self):
        print("[INFO] evaluation network")

        with torch.no_grad():
            self.model.eval()

            preds = []
            for (x,y) in self.testDataLoader:
                x = x.to(device)

                pred = self.model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())

        print(classification_report(self.testData.targets.cpu().numpy(),
        np.array(preds), target_names= self.testData.classes))

    def plot(self):
        print("[INFO] Saving plot")

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.H["train_loss"], label="train_loss")
        plt.plot(self.H["val_loss"], label="val_loss")
        plt.plot(self.H["train_acc"], label="train_acc")
        plt.plot(self.H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args.plot)

    def save_model(self):
        print("[INFO] Saving model")

        torch.save(self.model, args.model)

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", 
                    "--model", 
                    type=str,
                    required=True, 
                    help="path to output trained model")
    ap.add_argument("-p",
                    "--plot", 
                    type=str, 
                    required=True, 
                    help="path to output loss/accuracy plot")
    ap.add_argument("--init_lr", type=int, default=1e-3)
    ap.add_argument("--BATCH_SIZE", type=int, default=64)
    ap.add_argument("--EPOCH", type=int, default=10)
    ap.add_argument("--TRAIN_SPLIT", type=int, default=0.75)
    ap.add_argument("--VAL_SPLIT", type=int, default=0.25)
    ap.add_argument("--get_device", default="cuda")
    ap.add_argument("--use_plot", action="store_true")
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_parser()
    trainer = Trainer(args)
    trainer.DataLoading()
    trainer.modelLoader()
    trainer.training()
    trainer.eval()
    if args.use_plot:
        trainer.plot()
    trainer.save_model()
    