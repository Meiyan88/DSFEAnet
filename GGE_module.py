import torch_geometric.nn as tnn
import torch.nn as nn
import torch
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.nn.pool.edge_pool import EdgePooling
import torch.nn.functional as F

class ResGCN(nn.Module):
    def __init__(self, gcn_params=None, time=None):
        super(ResGCN, self).__init__()

        ### Encoding
        self.sage1 = tnn.SAGEConv(gcn_params['in_channels'], gcn_params['out_channels1'], normalize=True)
        self.sage2 = tnn.SAGEConv(gcn_params['out_channels1'], gcn_params['out_channels2'], normalize=True)
        self.sage3 = tnn.SAGEConv(gcn_params['out_channels2'], gcn_params['out_channels3'], normalize=True)
        self.sage4 = tnn.SAGEConv(gcn_params['out_channels3'], gcn_params['out_channels4'], normalize=True)

        self.layer1 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels1'], gcn_params['out_channels1'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels1']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[0])
        self.layer2 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels2'], gcn_params['out_channels2'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels2']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[1])
        self.layer3 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels3'], gcn_params['out_channels3'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels3']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[2])
        self.layer4 = self.make_layer(
            conv=tnn.SAGEConv(gcn_params['out_channels4'], gcn_params['out_channels4'], normalize=True),
            norm=nn.BatchNorm1d(num_features=gcn_params['out_channels4']), act=nn.LeakyReLU(True),
            block='res+',
            time=time[3])

        self.tr_IDH = nn.Linear(gcn_params['out_channels4'], gcn_params['out_channels5'])
        self.tr_1P19Q = nn.Linear(gcn_params['out_channels4'], gcn_params['out_channels5'])
        self.drop = torch.nn.Dropout(p=gcn_params['dropout'])
        self.bano1 = nn.BatchNorm1d(num_features=gcn_params['out_channels1'])
        self.bano2 = nn.BatchNorm1d(num_features=gcn_params['out_channels2'])
        self.bano3 = nn.BatchNorm1d(num_features=gcn_params['out_channels3'])
        self.bano4 = nn.BatchNorm1d(num_features=gcn_params['out_channels4'])
        self.bano_IDH = nn.BatchNorm1d(num_features=gcn_params['out_channels5'])
        self.bano_1P19Q= nn.BatchNorm1d(num_features=gcn_params['out_channels5'])

        self.edge1 = EdgePooling(gcn_params['out_channels1'], edge_score_method=None, add_to_edge_score=0.5)
        self.edge2 = EdgePooling(gcn_params['out_channels2'], edge_score_method=None, add_to_edge_score=0.5)
        self.edge3 = EdgePooling(gcn_params['out_channels3'], edge_score_method=None, add_to_edge_score=0.5)

    def make_layer(self, conv, norm, act, block, time):
        layer = []
        for i in range(time):
            layer.append(DeepGCNLayer(conv=conv,
                                      norm=norm, act=act, block=block))
        return nn.ModuleList(layer)

    def encode(self, x, adj, batch):
        hidden1 = self.sage1(x, adj)
        hidden1 = self.bano1(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer1)):
            hidden1 = self.layer1[i](hidden1, adj)
        hidden1, edge_index, batch, _ = self.edge1(hidden1, adj, batch)

        ### 2
        hidden1 = self.sage2(hidden1, edge_index)
        hidden1 = self.bano2(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer2)):
            hidden1 = self.layer2[i](hidden1, edge_index)
        hidden1, edge_index, batch, _ = self.edge2(hidden1, edge_index, batch)

        ### 3
        hidden1 = self.sage3(hidden1, edge_index)
        hidden1 = self.bano3(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer3)):
            hidden1 = self.layer3[i](hidden1, edge_index)
        hidden1, edge_index, batch, _ = self.edge3(hidden1, edge_index, batch)

        ### 4
        hidden1 = self.sage4(hidden1, edge_index)
        hidden1 = self.bano4(hidden1)
        hidden1 = F.leaky_relu(hidden1)
        for i in range(len(self.layer4)):
            hidden1 = self.layer4[i](hidden1, edge_index)
        slim = tnn.global_add_pool(hidden1, batch)

        slim_IDH = self.tr_IDH(slim)
        slim_IDH =self.bano_IDH(slim_IDH)
        slim_IDH = F.leaky_relu(slim_IDH)

        slim_1P19Q = self.tr_1P19Q(slim)
        slim_1P19Q = self.bano_IDH(slim_1P19Q)
        slim_1P19Q = F.leaky_relu(slim_1P19Q)

        return slim_IDH,slim_1P19Q,slim

    def forward(self, x, adj, batch):
        feature_GCN_x = self.encode(x, adj, batch)  ## mu, log sigma
        return feature_GCN_x
