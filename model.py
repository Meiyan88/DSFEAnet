import torch.nn as nn
import torch
import torch.nn.functional as F
from GGE_module import ResGCN
from DSFA_module import ContrastiveLoss,geometric_consistency_loss
from MMFE_module import mismatch_resnet

class PKAFnet(nn.Module):
    def __init__(self,  GCNencoder_name='resnet18',int_channel=1, classes=1, gcn_params=None, p=0.4,args=None):
        super(PKAFnet, self).__init__()
        self.config=args
        self.GCNencoder_name = GCNencoder_name
        self.encode_t1 = ResGCN(gcn_params, time=[2, 2, 2, 2]).to(self.config.device[0])
        self.encode_mismatch_1 =mismatch_resnet().to(self.config.device[0])
        self.encode_mismatch_2 = mismatch_resnet().to(self.config.device[1])

        self.feature_match_1=ContrastiveLoss().to(self.config.device[1])
        self.feature_match_2 = ContrastiveLoss().to(self.config.device[1])

        self.feature_consistency_1 = geometric_consistency_loss().to(self.config.device[1])
        self.feature_consistency_2 = geometric_consistency_loss().to(self.config.device[1])

        self.c_feature12 = nn.Linear(1024*2, 256).to(self.config.device[1])
        self.c_feature1 = nn.Linear(1024, 256).to(self.config.device[1])
        self.c_feature2 = nn.Linear(1024, 256).to(self.config.device[1])
        self.c_feature3 = nn.Linear(512, 256).to(self.config.device[1])

        self.c2_IDH = nn.Linear(256*3 , gcn_params['cate_class']).to(self.config.device[1])
        self.c2_1p19q = nn.Linear(256*3, gcn_params['cate_class']).to(self.config.device[1])

        self.drop_IDH = nn.Dropout(p=p).to(self.config.device[1])
        self.drop_1p19q = nn.Dropout(p=p).to(self.config.device[1])
        self.criterion_classifer = torch.nn.BCELoss()

    def forward(self,gcndata,image,T1,T1C,T2,FLAIR):
        x_IDH_1, x_1p19q_1, orth_score_1,feature1 = self.encode_mismatch_1(T1, T1C)
        x_IDH_2, x_1p19q_2, orth_score_2,feature2  = self.encode_mismatch_2(T2.to(self.config.device[1]), FLAIR.to(self.config.device[1]))

        feature12 = torch.cat([feature1.to(self.config.device[1]), feature2], 1)
        feature12 = self.c_feature12(feature12.to(self.config.device[1]))
        feature12 = F.leaky_relu(feature12)

        feature1 = self.c_feature1(feature1.to(self.config.device[1]))
        feature1 = F.leaky_relu(feature1)

        feature2 = self.c_feature2(feature2.to(self.config.device[1]))
        feature2 = F.leaky_relu(feature2)

        x, adj, lengs = self.get_data_from_graph(gcndata)
        feature_IDH_2, feature_1p19q_2,feature3 = self.encode_t1(x, adj, lengs)  ## mu, log sigma

        feature3 = self.c_feature3(feature3.to(self.config.device[1]))
        feature3 = F.leaky_relu(feature3)

        loss_itm_1=self.feature_match_1(feature1.to(self.config.device[1]),feature2.to(self.config.device[1]))
        loss_itm_2 = self.feature_match_2(feature12.to(self.config.device[1]), feature3.to(self.config.device[1]))
        loss_itm=(loss_itm_1+loss_itm_2)/2

        loss_consistency_1 = self.feature_consistency_1(feature1.to(self.config.device[1]), feature2.to(self.config.device[1]))
        loss_consistency_2 = self.feature_consistency_2(feature12.to(self.config.device[1]), feature3.to(self.config.device[1]))

        loss_consistency = (loss_consistency_1 + loss_consistency_2)

        feature_IDH = torch.cat([x_IDH_1.to(self.config.device[1]), x_IDH_2, feature_IDH_2.to(self.config.device[1])], dim=1)
        feature_1p19q = torch.cat([x_1p19q_1.to(self.config.device[1]), x_1p19q_2.to(self.config.device[1]), feature_1p19q_2.to(self.config.device[1])], dim=1)
        result_IDH = self.c2_IDH(self.drop_IDH(feature_IDH.view(feature_IDH.size(0), -1)))
        result_1p19q = self.c2_1p19q(self.drop_1p19q(feature_1p19q.view(feature_1p19q.size(0), -1)))

        w_distance=loss_consistency.to(self.config.device[0])

        return torch.sigmoid(result_IDH).to(self.config.device[0]),torch.sigmoid(result_1p19q).to(self.config.device[0]),orth_score_1,orth_score_2.to(self.config.device[0]),w_distance,loss_itm.to(self.config.device[0])

    def get_data_from_graph(self, data):
        gra = data.x.to(self.config.device[0])
        adj = data.edge_index.to(self.config.device[0])
        batch = data.batch.to(self.config.device[0])
        return gra, adj, batch


