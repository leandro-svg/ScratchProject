import torch
import torchvision
import numpy
import sys
from model.LeandDetect import LeandroDetect
from LeandDetect_2 import LeandroDetect_2
from test_class import Test

class Visualizer():
    def __init__(self):
        super().__init__()

        self.threshold = 0.5
        self.use_cp = True
        print("Itinitializer of Visualizer")
    
    def average(self, a, b):
        c = (a+b)/2
        return c
    
    def make_threshold(self, a):
        if a > self.threshold:
            a = a
        else:
            a = 0
        return a


    

def average(a,b):
    c = (a+b)/2
    return c

if __name__ == '__main__':
    print(torch.__version__)
    print(torchvision.__version__)
    print(numpy.__version__)

    a  = 2
    b = 3
    average_c = average(a,b) 
    print(average_c)

    visualizer = Visualizer()
    c = visualizer.average(a,b)
    new_c = visualizer.make_threshold(c)
    print(new_c)

    detector = LeandroDetect()
    print("detector", detector)

    test = Test()
    test.is_working()
    leandrodetect2  = LeandroDetect_2()

    


    
