import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

class DGCNN(torch.nn.Module):
    def __init__(self, out_dim, k=9, aggr='mean'):
        super().__init__()
        out_dim=1

        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 128, 128, 128]), k, aggr)
        #self.conv2 = DynamicEdgeConv(MLP([2 * 32, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 4,64,64, 16]), 9, aggr)
        #self.lin1 = Linear(64 + 32, 512)
        self.lin2 = Linear(2*16,1)

        self.mlp = MLP([128, 64, 32, out_dim], dropout=0.5, norm=None)

    def forward(self, data):
        x = data.x
        batch = data.batch
        # print("Initital:",x.size())
        x1 = self.conv1(x, batch)
        # print("Conv1:",x1.size())
        #x2 = self.conv2(x1, batch)
        # print("Conv2:",x2.size())
        #out = self.lin1(torch.cat([x1, x2], dim=1))
        # print("Concat + lin:",out.size())
        #out = global_max_pool(out, batch)
        # print("Global max pool:",out.size())
        out = self.mlp(x1)
        out= (out-torch.mean(out))/(torch.std(out) + 0.00001)
        out = torch.sigmoid(out)
        print("Division:",out)
        xl = out * x
        xs = (1-out) * x
        xl = self.conv3(xl)
        xs = self.conv3(xs)
        xl = global_max_pool(xl,batch)
        xs = global_max_pool(xs,batch)
        #print("Xl:",xl.size())
        #print("Xs:",xs.size())
        mass = self.lin2(torch.cat([xl,xs],dim=1))
        #print(mass)
        #print("Outputnet:",out)
        #return F.log_softmax(out, dim=1).flatten()
        return mass.flatten()
