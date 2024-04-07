import os
import numpy as np
import torch.optim as optim
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from FireFormer.former_model import Former
import utils
import time
from LossFunction.focalloss import BCEFocalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size = 256
thrd=0.7

import config
con=config.Config

def train(model):
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = BCEFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    # optimizer = optim.Adam(model.parameters(), lr=4e-5)

    epoch=0
    t0=time.time()
    for i in range(1,100*200+1):
        # print(i)
        train_x,train_y=utils.sample_train(batch_size)
        ims = utils.nor2(train_x)  # ims/255
        ims = torch.Tensor(ims).cuda()
        optimizer.zero_grad()
        pred = model(ims)  # 8*9*101*101*1
        loss = criterion(pred, torch.Tensor(train_y).cuda())
        loss.backward()
        optimizer.step()

        if(i%100==0):
            t1=time.time()
            tp1,fp1,fn1=eval(model)
            t2=time.time()
            tp2, fp2, fn2=mytest(model)
            t3 = time.time()
            if(epoch==0):print('time:',t1-t0,t2-t1,t3-t2)
            else:print(epoch)
            print('eval',tp1, fp1, fn1)
            if(tp1+fp1>0 and tp1+fn1>0):
                p=tp1/(tp1+fp1)
                r=tp1/(tp1+fn1)
                if(p+r>0):
                    f1=2*p*r/(p+r)
                    fa=1-p
                    print('f1:',f1,p,r,fa)
            print('test', tp2, fp2, fn2)
            if (tp2 + fp2 > 0 and tp2 + fn2 > 0):
                p = tp2 / (tp2 + fp2)
                r = tp2 / (tp2 + fn2)
                if (p + r > 0):
                    f1 = 2 * p * r / (p + r)
                    fa=1-p
                    print('f1:',f1,p, r,fa)

                    # if (f1 > 0.90):
                    #     # path = '/home/ices/Fire/MODIS/model/UFormer/' + str(thrd) + '_' +'layer3_' + str(epoch) + '.pth'
                    #     path = '/home/ices/Fire/MODIS/model/UFormer/' + str(thrd) + '_' + 'layer4_' + str(epoch) + '.pth'
                    #     torch.save(model.state_dict(), path)


            t0 = time.time()
            epoch += 1

def eval(model):
    model.eval()
    tp=0
    fp=0
    fn=0

    index=0
    eva_num=con.eva_batch_num
    # thrd=0.7
    with torch.no_grad():
        while(index<eva_num):
            test_x, test_y = utils.sample_eva(index)
            ims = utils.nor2(test_x)  # ims/255
            input = torch.Tensor(ims).cuda()
            pred = model(input)  # 8*9*101*101*1
            pred=pred.detach().cpu().numpy()
            # pred=pred[:,:,:test_x.shape[2],:test_x.shape[3]]
            pred[pred<=thrd]=0
            pred[pred>thrd]=1

            tpp,fpp,fnn=utils.cal_hit(pred,test_y)
            tp+=tpp
            fp+=fpp
            fn+=fnn

            index+=1

    model.train()
    return tp,fp,fn

def mytest(model):
    model.eval()
    tp=0
    fp=0
    fn=0

    index=0
    test_num=con.test_batch_num
    # thrd=0.7
    with torch.no_grad():
        while(index<test_num):
            test_x, test_y = utils.sample_test(index)
            ims = utils.nor2(test_x)  # ims/255
            input = torch.Tensor(ims).cuda()
            pred = model(input)  # 8*9*101*101*1
            pred=pred.detach().cpu().numpy()
            # pred=pred[:,:,:test_x.shape[2],:test_x.shape[3]]
            pred[pred<=thrd]=0
            pred[pred>thrd]=1

            tpp,fpp,fnn=utils.cal_hit(pred,test_y)
            tp+=tpp
            fp+=fpp
            fn+=fnn

            index+=1

    model.train()
    return tp,fp,fn

if __name__ == '__main__':
    model = Former(6)
    # state_dict = torch.load('/home/ices/Fire/MODIS/model/UFormer/' + str(thrd) + '_' +'layer3_' + str(196) + '.pth')
    # state_dict = torch.load('/home/ices/Fire/MODIS/model/UFormer/' + str(0.65) + '_' + 'layer4_' + str(191) + '.pth')
    # model.load_state_dict(state_dict)
    model.cuda()
    train(model)