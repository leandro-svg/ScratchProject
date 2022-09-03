import torch
import torchvision
import numpy


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

    
