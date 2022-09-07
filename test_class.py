import numpy
class Test():
    def __init__(self):
        super().__init__()
        self.test = True
        print("We reached it")

    def is_working(self):
        print("Yeah it is.")

if __name__ == '__main__':
    print(numpy.__version__)

