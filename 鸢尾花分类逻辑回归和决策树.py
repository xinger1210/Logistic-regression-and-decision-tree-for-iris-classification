# -*- encoding = utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split#拆分数据集
from sklearn.preprocessing import StandardScaler#归一化
from sklearn.decomposition import PCA#降维
from sklearn.pipeline import Pipeline#封装流程
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics#度量性能
from sklearn.model_selection import cross_val_score#交叉验证
from sklearn.model_selection import GridSearchCV#网格搜索调参
iris_data=pd.read_csv('iris.csv')#读取数据
# print(iris_data.head())
# print(iris_data.isnull().any())#查看有没有哪里列里面有空值，结果是没有
#
# print(iris_data.duplicated().all())#看看有没有重复值，结果是没有
# print(pd.DataFrame(iris_data.groupby('Species'))[0])#查看鸢尾花有哪几种种类
#把鸢尾花的种类转换成数字
iris_data.replace(['Iris-setosa'],0,inplace=True)
iris_data.replace(['Iris-versicolor'],1,inplace=True)
iris_data.replace(['Iris-virginica'],2,inplace=True)
# print(iris_data.head())
iris_data=np.array(iris_data)#转换成矩阵格式
# print(iris_data[:10,:])#输出前十行看看

x,y=np.split(iris_data,(4,),axis=1)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)#拆分训练集，测试集
# print(x_train.shape)
# # print(y_train.shape)
y_train=y_train.reshape(-1,1)
'''
逻辑回归模型搭建训练
Pipeline(steps) 利用sklearn提供的管道机制Pipeline来实现对全部步骤的流式化封装与管理。
第一个环节：可以先进行 数据标准化 StandardScaler()
中间环节：可以加上 PCA降维处理 取2个重要特征
最终环节：逻辑回归分类器
'''
#封装一个Pipelin
pipe_LR_logic = Pipeline([
                    ('sc', StandardScaler()),#归一化
                    ('pca', PCA(n_components = 2)),#降维化
                    ('clf_lr', LogisticRegression(random_state=1,solver='liblinear',multi_class='auto'))#逻辑回归
                    ])
pipe_LR_logic.fit(x_train, y_train)# 开始训练
Pipeline(memory=None,
         steps=[('sc',
                 StandardScaler(copy=True, with_mean=True, with_std=True)),
                ('pca',
                 PCA(copy=True, iterated_power='auto', n_components=2,
                     random_state=None, svd_solver='auto', tol=0.0,
                     whiten=False)),
                ('clf_lr',
                 LogisticRegression(C=1.0, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=None, max_iter=100,
                                    multi_class='auto', n_jobs=None,
                                    penalty='l2', random_state=1,
                                    solver='liblinear', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False)
#分类器准确率评估
print("在训练集上的准确率：%0.4f" %pipe_LR_logic.score(x_train, y_train))
y_predict = pipe_LR_logic.predict(x_test)#将测试集数据喂到模型里面得到预测结果
accuracy = metrics.accuracy_score(y_test, y_predict)#计算预测结果的准确率
print("在测试集上的准确率：%0.4f" % accuracy)
target_names = ['setosa', 'versicolor', 'virginica']
print(metrics.classification_report(y_test, y_predict, target_names = target_names))#生成分类报告
# #用交叉验证机验证模型性能
iris_data=x
iris_target=y
scores = cross_val_score(pipe_LR_logic, iris_data, iris_target.ravel(), cv = 5,scoring='f1_macro')
print("5折交叉验证逻辑回归分类器的准确率：%.4f 误差范围：(+/- %.4f)"%(scores.mean(), scores.std()*2))

# 网格搜索验证调参
X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_target, random_state=0)
#逻辑回归有一个超参数C，调节超参数C提高模型精确度
param_range = [0.01,0.1, 1, 10, 100]     # 超参数集合
param_grid_lr= {'C': param_range,        # 正则化系数λ的倒数,数值越小，正则化越强
                'penalty': ['l1','l2']}  # 对参数的惩罚项(约束),增强泛化能力，防止overfit
# 创建 grid search实例
clf = GridSearchCV(estimator = LogisticRegression(random_state=0,solver='liblinear',multi_class ='auto',max_iter=1000), # 模型
                    param_grid = param_grid_lr,#模型需要实验的参数
                    scoring = 'accuracy',#评分标准是准确率
                    cv = 10,#10折交叉验证
                    iid=True)
best_model_logic = clf.fit(X_train,Y_train.ravel())#调好参后再训练一次获得最好的模型

# 查看效果最好的超参数
print("最好模型的超参数：")
print('Best Penalty:', best_model_logic.best_estimator_.get_params()['penalty'])
print('Best C:', best_model_logic.best_estimator_.get_params()['C'])
print('逻辑回归模型best score:%.10f' % best_model_logic.best_score_)
print("测试集准确率: %0.10f" %best_model_logic.score(X_test, Y_test))
'''
上面是逻辑回归模型的搭建，训练和调参
下面是决策树模型的搭建，训练和调参
'''
#封装一个Pipelin
pipe_LR_tree = Pipeline([
                    ('sc', StandardScaler()),#归一化
                    ('pca', PCA(n_components = 2)),#降维化
                    ('clf_lr', DecisionTreeClassifier())#逻辑回归
                    ])
pipe_LR_tree.fit(x_train, y_train)# 开始训练
#分类器准确率评估
print("在训练集上的准确率：%0.4f" %pipe_LR_tree.score(x_train, y_train))
y_predict = pipe_LR_tree.predict(x_test)#将测试集数据喂到模型里面得到预测结果
accuracy = metrics.accuracy_score(y_test, y_predict)#计算预测结果的准确率
print("在测试集上的准确率：%0.4f" % accuracy)
target_names = ['setosa', 'versicolor', 'virginica']
print(metrics.classification_report(y_test, y_predict, target_names = target_names))#生成分类报告
#用交叉验证机验证模型性能
scores = cross_val_score(pipe_LR_tree, iris_data, iris_target.ravel(), cv = 5,scoring='f1_macro')
print("5折交叉验证逻辑回归分类器的准确率：%.4f 误差范围：(+/- %.4f)"%(scores.mean(), scores.std()*2))
#5折交叉验证逻辑回归分类器的准确率：0.8789 误差范围：(+/- 0.1500)
max_depth = range(1,10,1)
min_samples_leaf = range(1,10,2)
tuned_parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
# 创建 grid search实例
clf_tree = GridSearchCV(estimator = DecisionTreeClassifier(), # 模型
                    param_grid = tuned_parameters,#模型需要实验的参数
                    scoring = 'accuracy',#评分标准是准确率
                    cv = 10,#10折交叉验证
                    iid=True)
best_model_tree = clf_tree.fit(X_train,Y_train.ravel())#调好参后再训练一次获得最好的模型

# 查看效果最好的超参数
print("最好模型的超参数：")
print('Best max_depth:', best_model_tree.best_estimator_.get_params()['max_depth'])
print('Best min_samples_leaf:', best_model_tree.best_estimator_.get_params()['min_samples_leaf'])
print('决策树模型best score:%.10f' % best_model_tree.best_score_)
print("测试集准确率: %0.10f" %best_model_tree.score(X_test, Y_test))


