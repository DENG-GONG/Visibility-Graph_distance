# -*- coding: utf-8 -*-
# @Time : 2022/11/3 9:41
# @Author : DENG GQ
# @Email : 2832885485@qq.com
# @File : mlp_cluster.py
# @Project : Visibility-Graph_distance
import os

import numpy as np
import pandas as pd
import multiprocessing as mlp
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
from _tools import *
if os.path.exists('save_data/dataset.txt'):
    with open('save_data/dataset.txt',"r") as f:
        dataset=f.read().split(" ")
else:
    dataset=[]
    # res_pd=pd.DataFrame(None,columns=['dataset','vg','wvg','度序列距离','欧式距离','曼哈顿距离','皮尔森距离'])
    for file_path in os.listdir('../UCRArchive_2018'):
        try:
            train_data = pd.read_table(f"../UCRArchive_2018/{file_path}/{file_path}_TRAIN.tsv", header=None)
            test_data = pd.read_table(f"../UCRArchive_2018/{file_path}/{file_path}_test.tsv", header=None)
            if not train_data.isnull().any().any() and not test_data.isnull().any().any() and test_data.shape[1] == \
                    train_data.shape[1] and test_data.shape[0] + train_data.shape[0] < 10000 and train_data.shape[1] < 2500:
                dataset.append(file_path)
        except:
            pass
    with open('save_data/dataset.txt',"x") as f:
        f.write(" ".join(dataset))
def make_train_dis_matrix(data: pd.DataFrame):
    n = data.shape[0]
    degree_mat, vg_mat, wvg_mat = np.eye(n), np.eye(n), np.eye(n)
    per_mat=np.eye(n)
    for r in range(n):
        for c in range(r):
            x,y=data.iloc[r, :],data.iloc[c, :]
            WVG1, WVG2 = fastWVG_to_adj(x), fastWVG_to_adj(y)
            dd, vd, wvd = from_WVG_adj_to_3_distance(WVG1, WVG2)
            per_dis=1-abs(pearsonr(x,y)[0])
            degree_mat[r, c], degree_mat[c, r] = dd, dd
            vg_mat[r, c], vg_mat[c, r] = vd, vd
            wvg_mat[r, c], wvg_mat[c, r] = wvd, wvd
            per_mat[r,c],per_mat[c,r]=per_dis,per_dis
    return degree_mat, vg_mat, wvg_mat,per_mat


def make_test_dis_matrix(train_data: pd.DataFrame, test_data: pd.DataFrame):
    n1, n2 = test_data.shape[0], train_data.shape[0]
    degree_mat, vg_mat, wvg_mat = np.zeros([n1, n2]), np.zeros([n1, n2]), np.zeros([n1, n2])
    per_mat=np.zeros([n1,n2])

    for r in range(n1):
        for c in range(n2):
            x,y=test_data.iloc[r, :],train_data.iloc[c, :]
            WVG1, WVG2 = fastWVG_to_adj(x), fastWVG_to_adj(y)
            dd, vd, wvd = from_WVG_adj_to_3_distance(WVG1, WVG2)
            per_dis=1-abs(pearsonr(x,y)[0])
            degree_mat[r, c] = dd
            vg_mat[r, c] = vd
            wvg_mat[r, c] = wvd
            per_mat[r,c]=per_dis
    return degree_mat, vg_mat, wvg_mat,per_mat
def unity_task(filepath):
    if os.path.exists(f'save_data/{filepath}.csv'):
        return pd.read_csv(f'save_data/{filepath}.csv',index_col=0)
    res=[filepath]
    train_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_TRAIN.tsv",header=None)
    test_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_test.tsv",header=None)
    train_y=train_data.loc[:,0]
    test_y=test_data.loc[:,0]


    train_dd_mat,train_vd_mat,train_wvd_mat,train_per_mat=make_train_dis_matrix(train_data.loc[:,1:])
    test_dd_mat,test_vd_mat,test_wvd_mat,test_per_mat=make_test_dis_matrix(train_data.loc[:,1:],test_data.loc[:,1:])


    degree_model=KNeighborsClassifier(n_neighbors=1,metric='precomputed')
    vg_model=KNeighborsClassifier(n_neighbors=1,metric='precomputed')
    wvg_model=KNeighborsClassifier(n_neighbors=1,metric='precomputed')
    per_model=KNeighborsClassifier(n_neighbors=1,metric='precomputed')

    vg_model.fit(train_vd_mat,train_y)
    res.append(vg_model.score(test_vd_mat,test_y))

    wvg_model.fit(train_wvd_mat,train_y)
    res.append(wvg_model.score(test_wvd_mat,test_y))

    degree_model.fit(train_dd_mat,train_y)
    res.append(degree_model.score(test_dd_mat,test_y))

    eu_model=KNeighborsClassifier(n_neighbors=1,p=2)
    eu_model.fit(train_data.loc[:,1:],train_y)
    res.append(eu_model.score(test_data.loc[:,1:],test_y))

    manha_model=KNeighborsClassifier(n_neighbors=1,p=1)
    manha_model.fit(train_data.loc[:,1:],train_y)
    res.append(manha_model.score(test_data.loc[:,1:],test_y))

    per_model.fit(train_per_mat,train_y)
    res.append(per_model.score(test_per_mat,test_y))

    res_pd = pd.DataFrame([res], columns=['dataset', 'vg', 'wvg', '度序列距离', '欧式距离', '曼哈顿距离', '皮尔森距离'])
    print(res_pd )
    res_pd.to_csv(f'save_data/cluster_res/{filepath}.csv')
    return res_pd

if __name__ == '__main__':
    print(dataset)
    print(len(dataset))
    pool=mlp.Pool(8)
    res=pd.concat(pool.map(unity_task,dataset),axis=0,ignore_index=True)
    res.to_csv("'savedata/cluster_res/all_res.csv")
    # res=[]
    # for f  in dataset[:1]:
    #     res.append(unity_task(f))
    # print(res)
    # print(pd.concat(res,axis=0,ignore_index=True))