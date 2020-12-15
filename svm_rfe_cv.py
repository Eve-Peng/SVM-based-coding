from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv(r'C:/Users/liyi/Desktop/1.csv',header=0,index_col=None,sep = ',')
y = data['label']
x = data.drop(labels = ['label'],axis = 1)
clf = SVC(kernel="linear", C = 1000,gamma=0.1)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5))
rfecv.fit(x, y)
print(rfecv.ranking_)

print(rfecv.grid_scores_)
plt.figure()
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
