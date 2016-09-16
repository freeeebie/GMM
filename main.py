import numpy as np
import scipy.io.wavfile
import sklearn
import talkbox.mfcc as mp
from sklearn.mixture import GMM
import os
import sys
from gmm.gmm import gmm

def searchfiles(dirname):
    filelist = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(('.wav')):
                filelist.append(file)
    return filelist

def getmfccfeat(file):
    mfccfeat = []
    sample_rate, X = scipy.io.wavfile.read(file)
    ceps, mspec, spec = mp.mfcc(X)
    np.save("mfcc_cache%s"%file, ceps)  # cache results so that ML becomes fast
    ceps = np.load("mfcc_cache%s.npy"%file)
    num_ceps = len(ceps)
    return np.array(ceps)

gmm_set = {}

#training file
trainging_files = searchfiles('./training_sounds')
for file in trainging_files:
    Vectors = getmfccfeat(file)
    g = GMM(n_components=4, covariance_type='full')
    X = g.fit(Vectors)
    gmm_set[file] = g

# classify
test_files = searchfiles('./testing_sounds')
for file in test_files:
    Vectors = getmfccfeat(file)
    bestscore = -sys.maxsize
    score = 0
    for label, gmm in gmm_set.items():
        score = gmm.score_samples(Vectors)[0]
        score = np.sum(score)
        if score > bestscore:
            bestscore = score
            bestlabel = label
    print("%s is similar to %s" % (file, bestlabel))
