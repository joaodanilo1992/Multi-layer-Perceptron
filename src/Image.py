import numpy as np
import matplotlib.pylab as plt

class Image(object):

    def __init__(self, row, col):
        self.imgTraining = np.zeros((row, col, 280))
        self.imgTest = np.zeros((row,col, 120))

        countTraining = 0
        countTest = 0
        for person in range(1,41):
            for photo in range(1,11):
                if photo < 8:
                    self.imgTraining[:,:,countTraining] = plt.imread("orl_faces/s"+str(person)+"/"+str(photo)+".pgm")
                    countTraining = countTraining + 1
                else:                    
                    self.imgTest[:,:,countTest] = plt.imread("orl_faces/s"+str(person)+"/"+str(photo)+".pgm")
                    countTest = countTest + 1

    def getImgs(self):
        return self.imgTraining, self.imgTest
