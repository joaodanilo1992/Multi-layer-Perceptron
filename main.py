from src.Image import Image
from src.D2PCA import D2PCA
import matplotlib.pylab as plt
import numpy as np
import itertools

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from scikitplot.metrics import plot_confusion_matrix


labels = {'1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p1',
          '0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p2',
          '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p3',
          '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p4',
          '0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p5',
          '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p6',
          '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p7',
          '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p8',
          '0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p9',
          '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p10',
          '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p11',
          '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p12',
          '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p13',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p14',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p15',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p16',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p17',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p18',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p19',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p20',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p21',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p22',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p23',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p24',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p25',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p26',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'p27',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0' : 'p28',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0' : 'p29',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0' : 'p30',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0' : 'p31',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0' : 'p32',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0' : 'p33',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0' : 'p34',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0' : 'p35',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0' : 'p36',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0' : 'p37',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0' : 'p38',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0' : 'p39',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1' : 'p40',
          '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0' : 'erro'
         }

class_names = ['erro', 's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
               's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',
               's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',
               's31','s32','s33','s34','s35','s36','s37','s38','s39','s40'
               ]

def listToString(list):
    s = str(list).strip('[]')
    s = s.replace(',', '')
    return s

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    np.savetxt('/confusion_matrix.csv', cm, delimiter=',')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

#-------------------------------------------------------------------------------

scaler = StandardScaler()

set = Image(112,92)

setTraining = set.getImgs()[0]
setTest = set.getImgs()[1]

dimReductionTraining = D2PCA(setTraining,10,10)
dimReductionTest = D2PCA(setTest,10,10)

setReducDimTraining = dimReductionTraining.getReducImg()
setReducDimTest = dimReductionTest.getReducImg()

for i in range(setReducDimTraining.shape[2]):
    setReducDimTraining[:,:,i] = normalize(setReducDimTraining[:,:,i])

for i in range(setReducDimTest.shape[2]):
    setReducDimTest[:,:,i] = normalize(setReducDimTest[:,:,i])

InputTraining = np.zeros((setReducDimTraining.shape[0]*setReducDimTraining.shape[1],setReducDimTraining.shape[2]))

for img in range(setReducDimTraining.shape[2]):
    InputTraining[:,img] = np.ravel(setReducDimTraining[:,:,img])

InputTest = np.zeros((setReducDimTest.shape[0]*setReducDimTest.shape[1],setReducDimTest.shape[2]))

for img in range(setReducDimTest.shape[2]):
    InputTest[:,img] = np.ravel(setReducDimTest[:,:,img])

targetTraining = np.zeros((40,280))

for index in range(40):
     targetTraining[index,7*index:7*(index+1)] = 1

targetTest = np.zeros((40,120))

for index in range(40):
     targetTest[index,3*index:3*(index+1)] = 1

mlp = MLPClassifier(hidden_layer_sizes=(30), max_iter=100000,
                    solver='sgd', verbose=True, tol=1e-4, random_state=1,
                    learning_rate_init=0.005, activation='tanh', shuffle=False,
                    momentum=0.0, alpha=0.05, learning_rate='constant',
                    nesterovs_momentum=False)

mlp.fit(InputTraining.T,targetTraining.T)

print("Training set score: %.1f%%" % (mlp.score(InputTraining.T, targetTraining.T)*100))
print("Test set score: %.1f%%" % (mlp.score(InputTest.T, targetTest.T)*100))
#
# pred = mlp.predict(InputTest)

# InputTrain = np.zeros((setReducDimTraining.shape[0]*setReducDimTraining.shape[1],280))
# InputTest = np.zeros((setReducDimTraining.shape[0]*setReducDimTraining.shape[1],120))
#
# countTraining = 0
# countTest = 0
# count = 0
# for person in range(1,41):
#     for photo in range(1,11):
#         if photo < 8:
#             InputTrain[:,countTraining] = Input[:,count]
#             countTraining = countTraining + 1
#         else:
#             InputTest[:,countTest] = Input[:,count]
#             countTest = countTest + 1
#         count = count + 1
# # test = np.zeros((setReducDimTest.shape[0]*setReducDimTest.shape[1],120))
# #
# # for img in range(setReducDimTest.shape[2]):
# #     test[:,img] = np.ravel(setReducDimTest[:,:,img])

#
# scaler.fit(InputTrain.T)
# InputTrain = scaler.transform(InputTrain.T)
#
# scaler.fit(InputTest.T)
# InputTest = scaler.transform(InputTest.T)
#

#

#
#
# targetTestLabel = []
# predicLabel = []
#
# for target in range(120):
#     aux = targetTest[:,target].astype(int).tolist()
#     targetTestLabel.append(labels[listToString(aux)])
#
# for target in range(120):
#     aux = pred.T[:,target].astype(int).tolist()
#
#     if listToString(aux) in labels:
#         predicLabel.append(labels[listToString(aux)])
#     else:
#         predicLabel.append(labels['0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'])
#
# #cm = confusion_matrix(targetTestLabel, predicLabel)
#
# #plt.figure(figsize=(20, 20))
# #plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')
# #plt.savefig('foo.eps', bbox_inches='tight')
# #plt.show()
