import numpy as np

class D2PCA(object):

    def __init__(self, images, vertDim, horDim):

        self.imgs = images
        self.vertDim = vertDim
        self.horDim = horDim
        self.meanImg = self.getMeanImage()
        self.Hor = self.getHor()
        self.Ver = self.getVer()
        self.u = self.getU()
        self.v = self.getV()


    def getMeanImage(self):
        meanImg = np.zeros((self.imgs.shape[0], self.imgs.shape[1]))
        for img in range(self.imgs.shape[2]):
            meanImg = meanImg + self.imgs[:,:,img]
        return meanImg/self.imgs.shape[2]

    def getHor(self):
        dimReducHor = np.zeros((self.imgs.shape[1], self.imgs.shape[1]))
        for img in range(self.imgs.shape[2]):
            dimReducHor = dimReducHor + ((self.imgs[:,:,img] - self.meanImg).T.dot(self.imgs[:,:,img] - self.meanImg))
        return dimReducHor/self.imgs.shape[2]

    def getU(self):
        eigValue, eigVector = np.linalg.eig(self.Hor)

        #np.savetxt('../valorGh.out', eigValue, delimiter=',')
        #np.savetxt('../vetorGh.out', eigVector, delimiter=',')

        eigValueOrd = -np.sort(-eigValue)
        eigValueOrd = eigValueOrd[0:self.horDim]
        u = np.zeros((eigVector.shape[0],self.horDim))
        for value in range(self.horDim):
            u[:,value] = eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]
        return u

    def getVer(self):
        dimReducVer = np.zeros((self.imgs.shape[0], self.imgs.shape[0]))
        for img in range(self.imgs.shape[2]):
            dimReducVer = dimReducVer + ((self.imgs[:,:,img] - self.meanImg).dot((self.imgs[:,:,img] - self.meanImg).T))
        return dimReducVer/self.imgs.shape[2]

    def getV(self):
        eigValue, eigVector = np.linalg.eig(self.Ver)

        #np.savetxt('../Gv.out', self.Ver, delimiter=',')
        #np.savetxt('../valorGv.out', eigValue, delimiter=',')
        #np.savetxt('../vetorGv.out', eigVector, delimiter=',')



        eigValueOrd = -np.sort(-eigValue)
        eigValueOrd = eigValueOrd[0:self.vertDim]

        v = np.zeros((eigVector.shape[0],self.vertDim))

        for value in range(self.vertDim):
            v[:,value] = eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]

        # for value in range(self.vertDim):
        #
        #     if value % 2 == 0:
        #         v[:,value] = -eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]
        #     else:
        #         v[:,value] = eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]
        #
        #     if value >= 8:
        #         if value % 2 == 0:
        #             v[:,value] = eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]
        #         else:
        #             v[:,value] = -eigVector[:,np.where(eigValue == eigValueOrd[value])[0][0]]

        return v

    def getReducImg(self):

        res = np.zeros((self.vertDim, self.horDim, self.imgs.shape[2]))
        # np.savetxt('../v.out', self.v, delimiter=',')
        # np.savetxt('../u.out', self.u, delimiter=',')
        # np.savetxt('../foto1.out', self.imgs[:,:,0], delimiter=',')
        for img in range(self.imgs.shape[2]):
            res[:,:,img] = (self.v.T).dot(self.imgs[:,:,img]).dot(self.u)
        return res
