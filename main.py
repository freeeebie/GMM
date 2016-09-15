import numpy as np
import scipy.io.wavfile
import sklearn
import talkbox.mfcc as mp
from sklearn.mixture import GMM
import os

def searchfiles(dirname):
    filelist = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            filelist.append(file)
    print(filelist)
    return filelist
def getmfccfeat(files):
    mfccfeat = []
    for file in files:
        sample_rate, X = scipy.io.wavfile.read(file)
        ceps, mspec, spec = mp.mfcc(X)
        np.save("mfcc_cache%s"%file, ceps)  # cache results so that ML becomes fast
        ceps = np.load("mfcc_cache%s.npy"%file)
        num_ceps = len(ceps)
        mfccfeat.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    # print(mfccfeat)
    return np.array(mfccfeat)


trainging_files = searchfiles('./training_sounds')
Vectors = getmfccfeat(trainging_files)
# use Vectors as input values vector for neural net, k-means, etc
g = GMM(n_components=4, covariance_type='full')
X = g.fit(Vectors)
print(X)
test_files = searchfiles('./testing_sounds')
predict = getmfccfeat(test_files)
print('predict')
print(g.predict(predict))

