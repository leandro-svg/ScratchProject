import matplotlib

from model.LeandDetect import LeandNet
matplotlib.use("Agg")
from model.ConvNet import ConvNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import KMNIST
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomRotation
from torch.optim import Adam, SGD
from torch import nn
import matplotlib.pyplot as plt
import numpy as numpy
import numpy as np
import argparse
import torch
import time
from utils.functions import  SavePath
from utils.coco import  CocoDetection, COCO_CLASSES, CutoutPIL, COCO_LABEL_MAP
import torch.optim as optim
from utils.loss_coco import AsymmetricLoss
CUDA_LAUNCH_BLOCKING=1

class Trainer():
    def __init__(self, args):
        super().__init__()
        print("Training begins...")
        if args.dataset is not None:
            self.dataset = args.dataset
        else : 
            self.dataset = "KMNIST"
        
        if torch.cuda.device_count() == 0:
            print('[INFO] No GPUs detected. Exiting...')

        if torch.cuda.is_available():
            if not args.not_cuda:
                print("[INFO] CUDA device will be used")
                #torch.set_default_tensor_type('torch.cuda.FloatTensor')
                self.device = torch.device("cuda")
            if  args.not_cuda:
                print("[INFO] You have a CUDA device but choose to use the CPU")
                self.device = torch.device("cpu")
                #torch.set_default_tensor_type('torch.FloatTensor')
        else:
            print("[INFO] You don't have CUDA device installed. Use of the CPU")
            self.device = torch.device("cpu")
            #torch.set_default_tensor_type('torch.FloatTensor')

        if args.autoscale and args.BATCH_SIZE != 8:
            factor = args.BATCH_SIZE/8
            self.lr = args.init_lr * factor
            self.max_iter = args.max_iter // factor
            self.lr_steps = [x // factor for x in args.lr_steps]
        else:
            self.lr = args.init_lr
            self.max_iter = args.max_iter
            self.lr_steps = args.lr_steps
        self.loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

    def DataLoading(self):

        
        if self.dataset == "KMNIST" : 
            print("[INFO] Loading the KMINST data...")
            transform = transforms.Compose([
                    ToTensor(),
                ])
            self.trainData = KMNIST(root="data", train=True, download=True, transform=transform)
            self.testData = KMNIST(root="data", train=False, download=True, transform=transform)
            print("[INFO] generating the train/validation split...")
            numTrainSamples = int(len(self.trainData)*args.TRAIN_SPLIT)
            numValSamples = int(len(self.trainData)*args.VAL_SPLIT)
            (self.trainData, valData) = random_split(self.trainData, 
                                    [numTrainSamples, numValSamples],
                                    generator = torch.Generator().manual_seed(42))
            self.trainDataLoader = DataLoader(self.trainData, shuffle=True, batch_size = args.BATCH_SIZE, num_workers=args.num_workers )
            self.valDataLoader = DataLoader(valData, shuffle=False, batch_size = args.BATCH_SIZE, num_workers=args.num_workers )
            self.testDataLoader = DataLoader(self.testData, shuffle=False, batch_size = args.BATCH_SIZE, num_workers=args.num_workers)

        elif self.dataset == "COCO":
            print("[INFO] Loading the COCO data...")
            if args.augment:
                transform = transforms.Compose([
                    RandomRotation(degrees=(0, 180)),
                    ToTensor(),
                    transforms.ColorJitter(brightness=.5, hue=.3),
                    transforms.Resize((448, 448))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    #CutoutPIL(cutout_factor=0.5),
                    ToTensor(),
                ])
            self.trainData = CocoDetection(root="./data/coco/images/train2017",
                            annFile="data/coco/annotations/instances_train2017.json",
                            transform=transform)
            self.testData = CocoDetection(root="./data/coco/images/test2017",
                            annFile="data/coco/images/image_info_test2017/annotations/image_info_test2017.json",
                            transform=transform)
            valData = CocoDetection(root="./data/coco/images/val2017",
                            annFile="data/coco/annotations/instances_val2017.json",
                            transform=transform)
            print("[INFO] generating the train/validation split...")

            self.trainDataLoader = DataLoader(self.trainData, shuffle=True, batch_size = args.BATCH_SIZE, num_workers=args.num_workers, collate_fn=lambda x: x )
            self.valDataLoader = DataLoader(valData, shuffle=False, batch_size = args.BATCH_SIZE, num_workers=args.num_workers, collate_fn=lambda x: x )
            self.testDataLoader = DataLoader(self.testData, shuffle=False, batch_size = args.BATCH_SIZE, num_workers=args.num_workers, collate_fn=lambda x: x )

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
        if self.dataset == "KMNIST" :
            print("[INFO] Using LeandNet...")
            self.model = LeandNet(num_channels=1, classes=len(self.trainData.dataset.classes))
        elif self.dataset == "COCO":
            print("[INFO] Using ConvNet...")
            self.model = ConvNet(num_channels=1, classes=len(COCO_CLASSES)+1) 
        #To be done : Be able to stop training, save weights and begin again later on

        if not args.not_cuda:
            self.model.to("cuda")
        else:
            self.model.to("cpu")
        if args.opt == "Adam":
                self.opt = Adam(self.model.parameters(), lr=self.lr, weight_decay = args.decay, amsgrad = args.amsgrad)
        elif args.opt == "SGD":
            self.opt = SGD(self.model.parameters(), lr=self.lr, weight_decay = args.decay, momentum = args.momentum)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min',
                                                             factor=args.lr_reduce_factor,
                                                             patience=args.lr_schedule_patience,
                                                             verbose=True, min_lr=args.min_lr)

        if args.lossFn == "NLL":
            self.lossFn = nn.NLLLoss()
        elif args.lossFn == "CrossEntropy":
            self.lossFn = nn.CrossEntropyLoss()

    def training(self):

        trainer.DataLoading()
        trainer.modelLoader()
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        save_path = lambda epoch, iteration: SavePath("model", epoch, iteration).get_path(root=args.interrupt)

        print("[INFO] training our neural network...")
        startTime = time.time()
        try:
            for e in range(0, args.EPOCH):
                self.model.train()

                totalTrainLoss = 0
                totalValLoss = 0

                trainCorrect = 0
                valCorrect = 0
                iteration = 0
                
                if self.dataset == "KMNIST":
                    print("[INFO] Training on KMNIST dataset")
                    for x,y in self.trainDataLoader:
                        #if iteration == args.max_iter:
                         #   break
                        
                        (x,y) = (x.to(self.device), y.to(self.device))
                        pred =  self.model(x)
                        loss = self.lossFn(pred, y)
                        self.opt.zero_grad()
                        loss.backward()
                        if torch.isfinite(loss).item():
                            #self.scheduler.step(loss)
                            self.opt.step()
                        totalTrainLoss += loss
                        trainCorrect += (pred.argmax(1) == y).type(
                            torch.float).sum().item()
                        iteration += 1
                elif self.dataset == "COCO":
                    print("[INFO] Training on COCO dataset")
                    for X in self.trainDataLoader:
                        if iteration == args.max_iter:
                            break
                        
                        label = torch.tensor([])
                        images = torch.tensor([])
                        iteration=0
                        for input in X:
                            iteration += 1
                            try:
                                label = torch.cat((label, torch.tensor([input[1][0]['category_id']])))
                                images = torch.cat((images, (input[0])))
                            except:
                                pass    

                        images = images.reshape([int(np.shape(images)[0]/3),3,448,448])
                        images = images[:,1,:,:].unsqueeze(0)
                        images = images.permute(1,0,2,3)

                        (x,y) = (images.to(self.device), label.to(self.device))
                        pred =  self.model(x)
                        #loss = criterion(pred, y)
                        y = (y.cpu().detach().numpy().astype(int))
                        for elem in range(len(y)):
                            y[elem] = COCO_LABEL_MAP[y[elem]]
                        y = torch.tensor(y).to(self.device)
                        loss = self.lossFn(pred, y)
                        
                        self.opt.zero_grad()
                        loss.backward()
                        if torch.isfinite(loss).item():
                             #self.scheduler.step(loss)
                             self.opt.step()
                
                #To be done : Data parrallelism
                        totalTrainLoss += loss
                        trainCorrect += (pred.argmax(1) == y).type(
                            torch.float).sum().item()
                        iteration += 1
                with torch.no_grad():
                    print("[INFO] Evaluating ...")
                    self.model.eval()

                    for (x,y) in self.valDataLoader:
                        if iteration == args.max_iter:
                            break
                        if self.dataset == "KMNIST":
                            (x,y) = (x.to(self.device), y.to(self.device))
                            pred =  self.model(x)
                            totalValLoss += self.lossFn(pred, y)

                        elif self.dataset == "COCO":
                            label = torch.tensor([])
                            images = torch.tensor([])
                            iteration=0
                            for input in X:
                                iteration += 1
                                try:
                                    label = torch.cat((label, torch.tensor([input[1][0]['category_id']])))
                                    images = torch.cat((images, (input[0])))
                                except:
                                    pass    

                            images = images.reshape([int(np.shape(images)[0]/3),3,448,448])
                            images = images[:,1,:,:].unsqueeze(0)
                            images = images.permute(1,0,2,3)

                            (x,y) = (images.to(self.device), label.to(self.device))
                            pred =  self.model(x)
                            y = (y.cpu().detach().numpy().astype(int))
                            for elem in range(len(y)):
                                y[elem] = COCO_LABEL_MAP[y[elem]]
                            y = torch.tensor(y).to(self.device)
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
                self.H["val_acc"].append(valCorrect)

                print("[INFO] EPOCH: {}/{}".format(e + 1, args.EPOCH))
                print("Train loss : {:.6f}, Train accuracay : {:.4f}".format(
                    avgTrainLoss, trainCorrect))
                print("Val loss : {:.6f}, Val accuracay : {:.4f}".format(
                    avgValLoss, valCorrect))

        except KeyboardInterrupt:
            if args.interrupt:
                print('[INFO] Stopping early. Saving network...')
                
                SavePath.remove_interrupt(args.interrupt)
                saved_path = save_path(e, repr(iteration) + '_interrupt')
                torch.save(self.model, saved_path)
            exit()
        endTime = time.time()
        print("[INFO] total time taken to trian the model : {:.2f}s".format(
        endTime - startTime))


        trainer.eval()

        if args.use_plot:
            trainer.plot()
        trainer.save_model()

    def eval(self):
        print("[INFO] evaluation network")

        with torch.no_grad():
            self.model.eval()

            preds = []
            for (x,y) in self.testDataLoader:
                x = x.to(self.device)

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
    ap.add_argument("--init_lr", type=float, default=1e-3)
    ap.add_argument("--BATCH_SIZE", type=int, default=64)
    ap.add_argument("--EPOCH", type=int, default=10)
    ap.add_argument("--TRAIN_SPLIT", type=float, default=0.75)
    ap.add_argument("--VAL_SPLIT", type=float, default=0.25)
    ap.add_argument("--max_iter", type=int, default=400000)
    ap.add_argument("--lr_steps", type=list, default=[280000, 360000, 400000])
    ap.add_argument("--get_device", default="cuda")
    ap.add_argument("--use_plot", action="store_true")
    ap.add_argument("--dataset", default=None, type=str)
    ap.add_argument("--autoscale", action="store_true")
    ap.add_argument("--not_cuda", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--amsgrad", action="store_true")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--opt", type=str, default="Adam")
    ap.add_argument("--lossFn", type=str, default="NLL")
    ap.add_argument("--interrupt", type=str, default="./output")
    ap.add_argument('--lr_reduce_factor', type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    ap.add_argument('--lr_schedule_patience', type=int, default=5, help="Please give a value for lr_schedule_patience")
    ap.add_argument('--min_lr', type=float, default=1e-6, help="Please give a value for min_lr")

    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    trainer = Trainer(args)
    trainer.training()
    
    