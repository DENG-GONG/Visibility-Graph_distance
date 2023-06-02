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
from dtw import *
from sklearn.preprocessing import StandardScaler
def make_train_dis_matrix(data: pd.DataFrame):
    n = data.shape[0]
    degree_mat, vg_mat, wvg_mat = np.eye(n), np.eye(n), np.eye(n)
    per_mat=np.eye(n)
    for r in range(n):
        for c in range(r):
            x,y=data.iloc[r, :],data.iloc[c, :]
            VG1, VG2 = fastVG_to_adj(x), fastVG_to_adj(y)
            dd, vd = from_VG_adj_to_vg_and_degree_distance(VG1, VG2)
            per_dis=pearsonr(x,y)[0]
            degree_mat[r, c], degree_mat[c, r] = dd, dd
            vg_mat[r, c], vg_mat[c, r] = vd, vd
            # wvg_mat[r, c], wvg_mat[c, r] = wvd, wvd
            per_mat[r,c],per_mat[c,r]=per_dis,per_dis
    return degree_mat, vg_mat, wvg_mat,per_mat.max()+1-per_mat

def dtw_make_train_dis_matrix(data: pd.DataFrame):
    n = data.shape[0]
    dtw_eu_mat, dtw_mah_mat = np.zeros([n,n]), np.zeros([n,n])
    for r in range(n):
        for c in range(r):
            x,y=data.iloc[r, :],data.iloc[c, :]
            dtw_eu=dtw(x,y,distance_only=True).distance
            dtw_ma=dtw(x,y,dist_method='mahalanobis',distance_only=True).distance
            dtw_eu_mat[r,c]=dtw_eu
            dtw_eu_mat[c,r]=dtw_eu
            dtw_mah_mat[r,c]=dtw_ma
            dtw_mah_mat[c,r]=dtw_ma

    # dtw_eu_mat=dtw_eu_mat+dtw_eu_mat.T
    # dtw_mah_mat=dtw_mah_mat+dtw_mah_mat.T
    return dtw_eu_mat,dtw_mah_mat
def make_test_dis_matrix(train_data: pd.DataFrame, test_data: pd.DataFrame):
    n1, n2 = test_data.shape[0], train_data.shape[0]
    degree_mat, vg_mat, wvg_mat = np.zeros([n1, n2]), np.zeros([n1, n2]), np.zeros([n1, n2])
    per_mat=np.zeros([n1,n2])
    for r in range(n1):
        for c in range(n2):
            x,y=test_data.iloc[r, :],train_data.iloc[c, :]
            VG1, VG2 = fastVG_to_adj(x), fastVG_to_adj(y)
            dd, vd = from_VG_adj_to_vg_and_degree_distance(VG1, VG2)
            per_dis=pearsonr(x,y)[0]
            degree_mat[r, c] = dd
            vg_mat[r, c] = vd
            # wvg_mat[r, c] = wvd
            per_mat[r,c]=per_dis
    return degree_mat, vg_mat,wvg_mat,per_mat.max()+1-per_mat
def dtw_make_test_dis_matrix(train_data: pd.DataFrame, test_data: pd.DataFrame):
    n1, n2 = test_data.shape[0], train_data.shape[0]
    dtw_eu_mat, dtw_mah_mat = np.zeros([n1, n2]), np.zeros([n1, n2])
    for r in range(n1):
        for c in range(n2):
            x,y=test_data.iloc[r, :],train_data.iloc[c, :]
            dtw_eu = dtw(x, y, distance_only=True).distance
            dtw_ma = dtw(x, y, dist_method='mahalanobis', distance_only=True).distance
            dtw_eu_mat[r, c] = dtw_eu
            dtw_mah_mat[r, c] = dtw_ma

    return dtw_eu_mat,dtw_mah_mat
def unity_task(filepath,n):
    print('doing',filepath)
    if os.path.exists(f'save_data/cluster_res/n={n}/{filepath}.csv'):
        print(filepath+' have done')
        return pd.read_csv(f'save_data/cluster_res/n={n}/{filepath}.csv',index_col=0,encoding='utf-8')
    if not os.path.exists(f'save_data/cluster_res/n={n}'):
        os.makedirs(f'save_data/cluster_res/n={n}')
    res=[filepath]
    train_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_TRAIN.tsv",header=None)
    test_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_test.tsv",header=None)
    train_y=train_data.loc[:,0].copy()
    test_y=test_data.loc[:,0].copy()

    if os.path.exists(f'save_data/dis_mat/{filepath}.npy'):
        ((train_dd_mat, train_vd_mat, train_wvd_mat, train_per_mat),
         (test_dd_mat, test_vd_mat, test_wvd_mat, test_per_mat))=np.load(f'save_data/dis_mat/{filepath}.npy',allow_pickle=True)
    else:

        # Standar_model=StandardScaler()
        std_train=train_data.loc[:,1:]
        # std_train=(std_train-std_train.mean(axis=0))/std_train.std(axis=0)
        std_test=test_data.loc[:,1:]
        train_dd_mat,train_vd_mat,train_wvd_mat,train_per_mat=make_train_dis_matrix(std_train)
        test_dd_mat,test_vd_mat,test_wvd_mat,test_per_mat=make_test_dis_matrix(std_train,std_test)
        # np.save(f'save_data/dis_mat/{filepath}.npy',np.asarray([[train_dd_mat,train_vd_mat,train_wvd_mat,train_per_mat],[test_dd_mat,test_vd_mat,test_wvd_mat,test_per_mat]],dtype=object))


    degree_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')
    vg_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')
    # wvg_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')
    per_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')

    vg_model.fit(train_vd_mat,train_y)
    res.append(vg_model.score(test_vd_mat,test_y))

    # wvg_model.fit(train_wvd_mat,train_y)
    # res.append(wvg_model.score(test_wvd_mat,test_y))

    degree_model.fit(train_dd_mat,train_y)
    res.append(degree_model.score(test_dd_mat,test_y))

    eu_model=KNeighborsClassifier(n_neighbors=n,p=2)
    eu_model.fit(train_data.loc[:,1:],train_y)
    res.append(eu_model.score(test_data.loc[:,1:],test_y))

    manha_model=KNeighborsClassifier(n_neighbors=n,p=1)
    manha_model.fit(train_data.loc[:,1:],train_y)
    res.append(manha_model.score(test_data.loc[:,1:],test_y))

    per_model.fit(train_per_mat,train_y)
    res.append(per_model.score(test_per_mat,test_y))

    res_pd = pd.DataFrame([res], columns=['dataset', 'vg', '度序列距离', '欧式距离', '曼哈顿距离', '皮尔森距离'])
    res_pd.to_csv(f'save_data/cluster_res/n={n}/{filepath}.csv',encoding='utf-8')
    print("-"*10 + filepath + ' have done')
    return res_pd

def dtw_unity_task(filepath,n):
    print(f'doing {filepath},k={n}')
    if os.path.exists(f'save_data/cluster_res/n={n}/{filepath}_dtw.csv'):
        print(filepath+' dtw have done')
        return pd.read_csv(f'save_data/cluster_res/n={n}/{filepath}_dtw.csv',index_col=0,encoding='utf-8')
    if not os.path.exists(f'save_data/cluster_res/n={n}'):
        os.makedirs(f'save_data/cluster_res/n={n}')
    res=[filepath]
    train_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_TRAIN.tsv",header=None)
    test_data=pd.read_table(f"../UCRArchive_2018/{filepath}/{filepath}_test.tsv",header=None)
    train_y=train_data.loc[:,0].copy()
    test_y=test_data.loc[:,0].copy()

    if os.path.exists(f'save_data/dis_mat/{filepath}_dtw.npy'):
        ((train_dtw_eu_mat, train_dtw_mah_mat),
         (test_dtw_eu_mat, test_dtw_mah_mat))=np.load(f'save_data/dis_mat/{filepath}_dtw.npy',allow_pickle=True)
    else:


        std_train=train_data.loc[:,1:]

        std_test=test_data.loc[:,1:]
        train_dtw_eu_mat,train_dtw_mah_mat=dtw_make_train_dis_matrix(std_train)
        test_dtw_eu_mat,test_dtw_mah_mat=dtw_make_test_dis_matrix(std_train,std_test)
        np.save(f'save_data/dis_mat/{filepath}_dtw.npy',np.asarray([[train_dtw_eu_mat,train_dtw_mah_mat],[test_dtw_eu_mat,test_dtw_mah_mat]],dtype=object))

    dtw_eu_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')
    dtw_mah_model=KNeighborsClassifier(n_neighbors=n,metric='precomputed')

    dtw_eu_model.fit(train_dtw_eu_mat,train_y)
    res.append(dtw_eu_model.score(test_dtw_eu_mat,test_y))

    dtw_mah_model.fit(train_dtw_mah_mat,train_y)
    res.append(dtw_mah_model.score(test_dtw_mah_mat,test_y))
    # print(res)
    res_pd = pd.DataFrame([res], columns=['dataset', 'dtw_eu', 'dtw_mahalanobis'])
    res_pd.to_csv(f'save_data/cluster_res/n={n}/{filepath}_dtw.csv',encoding='utf-8')
    print("-"*10 + filepath + ' dtw have done')
    return res_pd
if __name__ == '__main__':
    # if os.path.exists('save_data/dataset_info.csv'):
    #     dataset_info=pd.read_csv("save_data/dataset_info.csv",index_col=0)
    # else:
    #     dataset = []
    #     len_=[]
    #     counts=[]
    #     # res_pd=pd.DataFrame(None,columns=['dataset','vg','wvg','度序列距离','欧式距离','曼哈顿距离','皮尔森距离'])
    #     for file_path in os.listdir('../UCRArchive_2018'):
    #         try:
    #             train_data = pd.read_table(f"../UCRArchive_2018/{file_path}/{file_path}_TRAIN.tsv", header=None)
    #             test_data = pd.read_table(f"../UCRArchive_2018/{file_path}/{file_path}_test.tsv", header=None)
    #             if not train_data.isnull().any().any() and not test_data.isnull().any().any() and train_data.shape[1]==test_data.shape[1]:
    #                 dataset.append(file_path)
    #                 len_.append(train_data.shape[1])
    #                 counts.append(train_data.shape[0]+test_data.shape[0])
    #         except:
    #             pass
    #     dataset_info=pd.DataFrame({'dataset':dataset,"len":len_,"counts":counts})
    #     dataset_info.to_csv("save_data/dataset_info.csv")
    #
    # print(dataset_info)
    # dataset=dataset_info['dataset'][dataset_info["counts"]<2000]
    # pool=mlp.Pool(6)
    # res=pool.starmap(unity_task, zip(dataset, [1] * len(dataset)))
    # res=pool.starmap(unity_task,zip(dataset,[1]*len(dataset)))
    # pool.close()
    # pool.join()
    # print(res)
    # res2=pd.concat(res,axis=0,ignore_index=True)
    # print(res2)

    dataset_sum=pd.read_excel("汇总结果.xlsx",index_col=0)
    # print(dataset)
    dataset=dataset_sum['dataset'].tolist()
    pool=mlp.Pool(5)
    res=pool.starmap(dtw_unity_task, zip(dataset,[1]*len(dataset)))
    pool.close()
    pool.join()
    # res=[]
    # res.append(dtw_unity_task(dataset[0],1))
    print(res)
    res=pd.concat(res,axis=0,ignore_index=True)
    res.to_excel("dtw_汇总.xlsx")