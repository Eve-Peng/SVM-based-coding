import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


data = pd.read_csv("C:/Users/liyi/Desktop/1.csv",header = 0 ,sep = ",")
index = ['t%s'%i for i in range(1,125)]
datasets = {}
datasets['ID'] = list(data.columns)
datasets['ID'].remove("label")
datasets['feature'] = np.array(data[index])
datasets['label'] = np.array(data['label'])
#datasets = datasets.load_wine()

featureNames = datasets['ID']
feat,label = datasets['feature'],datasets['label']
scaler = preprocessing.StandardScaler()
inputVec = scaler.fit_transform(feat)
tmp,feat = inputVec.copy(),featureNames.copy()
rank = []
score = []
while(tmp.shape[1]):
	clf = LinearSVC()
	clf.fit(tmp,label)
	coef = clf.coef_
	print(coef)
	scores = np.sum(coef**2,axis=0)
	_id_ = np.argmin(scores)
	rank.append(feat[_id_])
	score.append(scores)
	feat.pop(_id_)
	tmp = np.delete(tmp,_id_,axis=1)
print(rank)
print(score)
