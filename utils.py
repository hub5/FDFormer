import numpy as np
import random
import os

import config

con=config.Config

def sample_train(batch_size):
    res=[]
    for i in range(batch_size):
        sample_index = random.randint(1, con.train_num-1)
        if(sample_index<150964 and sample_index>=150960):continue#9*9 数据坏了
        # if (sample_index < 117620 and sample_index >= 117609): continue  # 15*15 数据坏了
        dat=np.load(con.train_path+str(sample_index)+'.npy')
        dat=dat[con.channel_id,:,:]
        res.append(dat)
    res=np.asarray(res)
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    return x,y

def sample_eva(index):
    res = np.load(con.eva_patch_path + str(index)+'.npy')
    res=res[:,con.channel_id]
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    # y[y != 9] = 0
    # y[y == 9] = 1
    return x,y

def sample_test(index):
    res = np.load(con.test_patch_path + str(index)+'.npy')
    res=res[:,con.channel_id]
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    # y[y != 9] = 0
    # y[y == 9] = 1
    return x,y


def sample_eva_all(index):
    res = np.load(con.test_path + con.eva_path_npy[index])[np.newaxis,:,:,:]
    res=res[:,con.channel_id]
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    y[y != 9] = 0
    y[y == 9] = 1
    return x,y

def sample_test_all(index):
    res = np.load(con.test_path + con.test_path_npy[index])[np.newaxis,:,:,:]
    res=res[:,con.channel_id]
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    y[y != 9] = 0
    y[y == 9] = 1
    return x,y

# def sample_test(index,batch_size):
#     res=[]
#     for i in range(batch_size):
#         sample_index = index+i
#         dat=np.load(test_path+test_list[sample_index])
#         dat=dat[channel_id,:,:]
#         res.append(dat)
#     res=np.asarray(res)
#     x=res[:,:-1,:,:]
#     y=res[:,-1:,:,:]
#     y[y != 9] = 0
#     y[y == 9] = 1
#     return x,y
#
# def sample_test2(index):
#     res=np.load(test_path+test_list[index])
#     res=res[:,channel_id]
#     x=res[:,:-1,:,:]
#     y=res[:,-1:,:,:]
#     # y[y != 9] = 0
#     # y[y == 9] = 1
#     return x,y

def sample_old(index):
    test_path = '/home/ices/Fire/MODIS/data/2022_pack/'
    # print(len(os.listdir(test_path)))
    res = np.load(test_path + str(index)+'.npy')
    res = res[:, con.channel_id]
    x=res[:,:-1,:,:]
    y=res[:,-1:,:,:]
    # y[y != 9] = 0
    # y[y == 9] = 1
    return x,y
# sample_old(10)

def nor(x):
    r=(x-con.min)/(con.max-con.min)
    return r

def nor2(x):
    b,_,h,w=x.shape
    x_new=np.zeros((b,6,h,w))

    # b,c,h,w
    #[0,3,4,5]
    # x[:, 0] = (x[:, 0] - con.min0) / (con.max0 - con.min0)
    # x[:, 1] = (x[:, 1] - con.min3) / (con.max3 - con.min3)
    # x[:, 2] = (x[:, 2] - con.min4) / (con.max4 - con.min4)
    # x[:, 3] = (x[:, 3] - con.min5) / (con.max5 - con.min5)

    #[1,2,4,5]
    x_new[:, 0] = (x[:, 0] - con.min1) / (con.max1 - con.min1)
    x_new[:, 1] = (x[:, 1] - con.min2) / (con.max2 - con.min2)
    x_new[:, 2] = (x[:, 2] - con.min4) / (con.max4 - con.min4)
    x_new[:, 3] = (x[:, 3] - con.min5) / (con.max5 - con.min5)

    tmp14=x[:, 0]-x[:, 2]
    x_new[:,4]=(tmp14-con.min_14)/(con.max_14-con.min_14)

    tmp15 = x[:, 0] - x[:, 3]
    x_new[:, 5] = (tmp15 - con.min_15) / (con.max_15 - con.min_15)

    # tmp24 = x[:, 1] - x[:, 2]
    # x_new[:, 6] = (tmp24 - con.min_24) / (con.max_24 - con.min_24)
    #
    # tmp25 = x[:, 1] - x[:, 3]
    # x_new[:, 7] = (tmp25 - con.min_25) / (con.max_25 - con.min_25)


    # #[1,3]
    # x[:,0]=(x[:, 0] - con.min1) / (con.max1 - con.min1)
    # x[:,1]=(x[:, 1] - con.min3) / (con.max3 - con.min3)
    return x_new

def nor3(x):
    #b,c,h,w
    x[:,0] = (x[:,0]-con.min0) / (con.max0-con.min0)
    x[:,1] = (x[:,1] - con.min1) / (con.max1 - con.min1)
    x[:,2] = (x[:,2] - con.min2) / (con.max2 - con.min2)
    x[:,3] = (x[:,3] - con.min3) / (con.max3 - con.min3)
    x[:, 4] = (x[:, 4] - con.min4) / (con.max4 - con.min4)
    x[:, 5] = (x[:, 5] - con.min5) / (con.max5 - con.min5)
    return x

def cal_hit(pred,true):
    #0/1:b,c,h,w

    # print(pred.flatten().tolist().count(0),pred.flatten().tolist().count(1))
    # print(true.flatten().tolist().count(0), true.flatten().tolist().count(1))

    ##tp:p+t=2
    tp=pred+true
    tp=tp.flatten().tolist().count(2)

    ##fp:p-t=1
    fp=pred-true
    fp = fp.flatten().tolist().count(1)

    ##fn:p-t=-1
    fn = pred - true
    fn = fn.flatten().tolist().count(-1)

    return tp,fp,fn

