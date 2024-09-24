import torch
import torch.nn as nn
import numpy
import time
from extensions.chamfer_dist import chamfer_3DDist
from models.PointCAE_bak2 import test
from utils.config import cfg_from_yaml_file
import numpy as np
from math import cos, sin, pi, log2, sqrt
import os
from sklearn.cluster import KMeans
from multiprocessing import Process, Pool
import torchvision.models as models
from torchsummary import summary
import struct

class TestNet(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.calloss()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.encoder_channel = encoder_channel
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def calloss(self):
        self.loss = chamfer_3DDist()


    def getloss(self, data , center):
        dist1, dist2, idx1, idx2  = self.loss(data, center)
        B, G, _ = center.shape
        B, N, _ = data.shape
        idx1 = idx1.view(-1)
        neighborhoodinit = torch.empty(B * G, 1, self.encoder_channel)
        for i in range(G):
            tmp = torch.nonzero(idx1 == i).reshape(1, -1)
            tmp = tmp.reshape(-1)
            for j in range(B):
                tmp1 = torch.nonzero(tmp < (j + 1) * N).view(-1)
                tmp1 = torch.nonzero(j*N <= tmp[tmp1]).view(-1)
                neighborhood = (data.view(B * N, -1)[tmp[tmp1], :] - center[j, i, :].reshape(1, -1)).unsqueeze(0)
                feature = self.first_conv(neighborhood.transpose(2, 1))
                _, _, n = feature.shape
                feature_global = torch.max(feature, dim=2, keepdim=True)[0]
                feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
                feature = self.second_conv(feature)
                neighborhoodinit[j * G + i] = torch.max(feature, dim=2, keepdim=False)[0].squeeze()
        print(neighborhoodinit.reshape(B, G, self.encoder_channel).shape)

    def neighbor(self,data):
        data = torch.rand(2, 2048, 3).cuda()
        for i in range(3):
            pts = data[i].squeeze().numpy()
            pred = KMeans(n_clusters=32).fit(pts)
            center = torch.tensor(pred.cluster_centers_)
            labels = torch.tensor(pred.labels_)
            pts = data[i].squeeze()
            for j in range(32):
                label = torch.nonzero(labels == j).reshape(1, -1)
                neibor = (pts[label]-center[j]).unsqueeze(0)
                print(neibor.shape)



    def seed2(self):
        n = int(sqrt(64))
        folding_seed = torch.empty(64, 3).cuda()
        count = 0
        for i in range(int((n + 2) / 2)):
            r = sqrt(1 - (i / (n + 2) * 2) ** 2)
            z0 = i / (n + 2) * 2
            for j in range(n-1):
                if i == 0:
                    angle = j / (n - 1 ) + i / (n + 2) * 2
                    x0 = r * cos(angle * pi * 2)
                    y0 = r * sin(angle * pi * 2)
                    folding_seed[count] = torch.tensor([x0, y0, z0])
                    count = count + 1
                else:
                    angle = j / (n - 1) + i / (n + 2) * 2
                    x0 = r * cos(angle * pi * 2)
                    y0 = r * sin(angle * pi * 2)
                    folding_seed[count] = torch.tensor([x0, y0, z0])
                    count = count + 1
                    folding_seed[count] = torch.tensor([x0, y0, -z0])
                    count = count + 1
            folding_seed[count] = torch.tensor([0, 0, 1])
        return folding_seed


def process0(data,num):
        pts = data[num]
        pred = KMeans(n_clusters=32).fit(pts)
        center = torch.tensor(pred.cluster_centers_)
        labels = torch.tensor(pred.labels_)
        pts = torch.tensor(data[num])
        neighbors = torch.empty(32, 64, 3)
        for j in range(32):
            label = torch.nonzero(labels == j).reshape(1, -1)
            neighbor = (pts[label] - center[j])
            _, n, _ = neighbor.shape
            if n < 64:
                a = neighbor.repeat(1, int(64 / n), 1)
                neighbor = torch.cat((a, neighbor[:, 0:64 % n, :]), 1)
            elif n > 64:
                neighbor = neighbor[:, 0:64, :]
            neighbors[j] = neighbor
        return neighbors



def main():
    '''    data = torch.rand(32, 2048, 3).cuda()
    center = torch.rand(32,32,3).cuda()
    s = time.time()
    a = TestNet(384)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a.to(device)
    a.getloss(data,center)
    e = time.time()
    print("耗时: {:.2f}秒".format(e - s))
    '''

    # data = torch.rand(32, 2048, 3).cuda()
    # data = data.cpu().numpy()

    pts = np.loadtxt("test/3.txt")
    pts = pts[:,0:3]
    pred = KMeans(n_clusters=5).fit(pts)
    center = pred.cluster_centers_
    labels = pred.labels_
    labels = np.expand_dims(labels,axis=1)
    pts2 = np.concatenate((pts,labels),axis =1)
    np.savetxt('test/4.txt',pts2,fmt = "%.4f")

    '''
    for i in range(32):
        pts = data[i].squeeze().cpu().numpy()
        pred = KMeans(n_clusters=32).fit(pts)
        center = torch.tensor(pred.cluster_centers_)
        labels = torch.tensor(pred.labels_)
        pts = data[i].squeeze().cpu()
        for j in range(32):
            label = torch.nonzero(labels == j).reshape(1, -1)
            neighbor = (pts[label] - center[j])
            _,n,_ = neighbor.shape
            if n < 64:
                a = neighbor.repeat(1, int(64 / n), 1)
                neighbor =torch.cat((a,neighbor[:,0:64%n,:]),1)
            elif n > 64:
                neighbor = neighbor[:,0:64,:]

    list = os.listdir('demo_55/')
    for l in list:
        with open('demo_55/'+l, 'r', encoding='utf-8') as f:
            text = f.read()
            with open('demo/'+l, 'ab') as f2:
                sn = struct.pack("s", text)
                f2.write(sn)

    '''



    # s = time.time()
    # pool = Pool(processes = 32)
    #
    #
    # results = []
    # for i in range(32):
    #     result = pool.apply_async(process0, (data,i,))
    #     results.append(result)
    # pool.close()
    # pool.join()
    # neighbor = torch.empty(32, 32,64, 3)
    # for res in results:
    #     neighbor[i] = res.get().unsqueeze(0)
    # print(neighbor.shape)
    #
    #
    # e = time.time()
    # print("耗时: {:.2f}秒".format(e - s))


    # pretrained=True就可以使用预训练的模型

    # state_dict = torch.load('experiments/PointCAE/PCN_models/NewBaseline/ckpt-last.pth')
    # for param_tensor in state_dict['base_model']:
    #     with open('experiments/PointCAE/PCN_models/Kmeans32_2/ckpt-last.txt', 'a') as file0:
    #         print(state_dict[param_tensor], file=file0)

    #summary(net, (32,2048,3))
    # with open('experiments/PointCAE/PCN_models/NewBaseline/ckpt-last.txt', 'a') as file0:
    #    print(state_dict, file=file0)



if __name__ == '__main__':
    main()
    '''
    #cfg = cfg_from_yaml_file('cfgs/ShapeNet34_models/PointCAE.yaml')
    #print(cfg.total_bs)
    #test(cfg.model)
    color = torch.tensor([[230, 25, 75], [60, 180, 75], [255, 255, 25], [67, 99, 216], [245, 130, 49], [145, 30, 180],
                          [66, 212, 244], [240, 50, 230], [191, 239, 79], [250, 190, 212], [70, 153, 144],
                          [220, 190, 255]])
    print(color.shape)
    print(color[0].shape)
    a= torch.ones(4,3)
    n, _ = a.shape
    colors = color[0].repeat(n,1)
    print(colors.shape)
    result = np.array(torch.cat((a.cpu(), colors.cpu()), 1))
    print(result)

    a = TestNet(384)
    fd = a.seed2()
    pts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pts')
    pt = fd.squeeze().detach().cpu()
    result = np.array(pt)
    pt_path = os.path.join(pts_path, 'fd_pt' + '.txt', )
    with open(pt_path, 'wb') as f:
        np.savetxt(f, result, fmt='%.04f', delimiter='    ')
    '''