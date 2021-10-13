# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
#
#
# iris=load_iris()
# x=iris.data
# y=iris.target
# # print(x)
# # print(y)
#
# # print(x.shape)
# import pandas as pd
#
# df=pd.DataFrame(x)
# # print(df)
#
# pca=PCA(n_components=3)
# pca=pca.fit(x)
# x_dr=pca.transform(x)
# # print(x_dr)
# # print(x_dr.shape)
#
# # print(x_dr[y==0,0])
# colors=["red","black","orange"]
# plt.figure()
# for i in range(0,3):
#     plt.scatter(x_dr[y==i,0],
#                 x_dr[y==i,1],
#                 alpha=0.7,
#                 c=colors[i],
#                 label=iris.target_names[i]
#                 )
# plt.legend()
# plt.title("PCA of iris dataset")
# plt.show()
#
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())
#
#
# pca_line=PCA().fit(x)
# print(pca_line.explained_variance_)
# print(pca_line.explained_variance_ratio_)
# print(pca_line.explained_variance_ratio_.sum())
#
# import  numpy as np
# np.cumsum(pca_line.explained_variance_ratio_)
#
# plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
# plt.xlabel("1111111")
# plt.xticks([1,2,3,4])
# plt.ylabel("222222")
# plt.show()
#
from sklearn.datasets import fetch_lfw_people
from  sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

face=fetch_lfw_people(min_faces_per_person=60)
# print(face.images.shape)
# print(face.data.shape)
X=face.data
# print(X)






