# -*- coding: utf-8 -*-
# @Time : 2022/9/26 10:19
# @Author : DENG GQ
# @Email : 2832885485@qq.com
# @File : _tools.py
# @Project : Project
import numpy as np
from scipy.spatial.distance import euclidean

__all__=['convert_2_row_vector','fastVG','fastWVG','VG_dis_between_two_TS','WVG_dis_between_two_TS','degree_dis_between_two_TS']
def convert_2_row_vector(TS) ->np.ndarray:
    """
    将能转为行向量的转为行向量
    :param TS:时间序列
    :return:行向量版TS
    """
    if not isinstance(TS,np.ndarray):
        TS=np.asarray(TS)
    _shape = TS.shape
    if TS.ndim == 1:
        return TS.reshape([1, _shape[0]])
    if TS.ndim == 2:
        if _shape[0] == 1:
            return TS
        if _shape[1] == 1:
            return TS.T
    raise f"仅允许输入能转为行向量的时间序列,而目前输入了shape={TS.shape}"
def fastVG(TS: np.ndarray, window: int):
    """
    内部函数,用于生成TS的有限视距的VG序列
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """

    _len = TS.shape[1]
    res = np.full((_len - 1, window - 1), False)
    max_k = np.full([1, _len], np.NINF)
    for i in range(1, window):
        k = (TS[0, i:] - TS[0, :-i]) / i
        index = k > max_k[0,:-i]
        max_k[0,np.where(index)[0]] = k[index]
        res[np.where(index)[0], i - 1] = True
    return res[:-1,1:]
def fastWVG(TS: np.ndarray, window: int):
    """
    内部函数,用于生成TS的有限视距的VG序列
    通过错位相减等向量计算法加速VG序列的生成
    变种邻接矩阵：列索引为向后多少位，即（i，j）表示节点i与节点i+j+1是否存在连边关系，由于VG网络里节点i与i+1一定存在连边关系，所以节省开销返回的矩阵表示i与i+j+2是否存在连边
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """

    _len = TS.shape[1]
    res = np.full((_len - 1, window - 1), 0)
    max_k = np.full([1, _len], np.NINF)
    for i in range(1, window):
        k = (TS[0, i:] - TS[0, :-i]) / i
        index = k > max_k[0,:-i]
        max_k[0,np.where(index)[0]] = k[index]

        res[np.where(np.all([k>=0,index],axis=0))[0], i - 1] = 1
        res[np.where(np.all([k<0,index],axis=0))[0], i - 1] = -1

    return res
def fastVG_to_norm_adjust_matrix(TS: np.ndarray, window: int):
    """
    用于生成TS的有限视距的VG网络
    通过错位相减等向量计算法加速VG序列的生成
    :param TS: 能转为行向量的时间序列
    :return: (_len-2)*(window-2)的变种邻接矩阵A,A的i,j处的值:表示ts的VG网络中节点i与i+j+2是否存在连边，True为有
    """

    _len = TS.shape[1]
    res = np.full([window,window],np.False_)
    max_k = np.full([1, _len], np.NINF)
    for i in range(1, window):
        k = (TS[0, i:] - TS[0, :-i]) / i
        index = k > max_k[0,:-i]
        where_is_true=np.where(index)[0]
        max_k[0,where_is_true] = k[index]
        for j in where_is_true:
            res[j,j+i]=True
            res[j+i,j]=True
    return res
def VG_dis_between_two_TS(TS1, TS2):
    """
    等长序列间的VG距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2=fastVG(TS1,TS1.shape[1]),fastVG(TS2,TS2.shape[1])
    return np.sum(VG1!=VG2)
def WVG_dis_between_two_TS(TS1, TS2):
    """
    等长序列间的VG距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2=fastWVG(TS1,TS1.shape[1]),fastWVG(TS2,TS2.shape[1])
    return np.sum(VG1!=VG2)
def degree_dis_between_two_TS(TS1,TS2):
    """
    等长序列间的VG度序列距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    VG1,VG2=fastVG_to_norm_adjust_matrix(TS1,TS1.shape[1]),fastVG_to_norm_adjust_matrix(TS2,TS2.shape[1])
    return euclidean(np.sum(VG1,axis=1),np.sum(VG2,axis=1))
def norm_adjust_to_new_adjust(norm_matrix:np.ndarray):
    m=norm_matrix.shape[0]
def new_adjust_to_norm_adjust(new_matrix:np.ndarray):
    m=new_matrix.shape[0]+1


def VG_dis_between_two_TS_for_3(TS1, TS2):
    """
    等长序列间的VG距离
    :param TS1:
    :param TS2:
    :return:
    """
    TS1,TS2=convert_2_row_vector(TS1),convert_2_row_vector(TS2)
    WVG1,WVG2=fastWVG(TS1,TS1.shape[1]),fastWVG(TS2,TS2.shape[1])
    VG1,VG2=abs(WVG1),abs(WVG2)
    return np.sum(VG1!=VG2),np.sum(WVG1!=WVG2)