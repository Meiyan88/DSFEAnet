import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=8, device='cuda:6', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵


    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到

        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class geometric_consistency_loss(nn.Module):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
    def forward(self, image_embeds, text_embeds):

        image_embeds = image_embeds
        text_embeds = text_embeds
        image_embeds = F.normalize(image_embeds, dim=1)  # (bs, dim)  --->  (bs, dim)
        text_embeds = F.normalize(text_embeds, dim=1)  # (bs, dim)  --->  (bs, dim)

        logits_text_per_image =  image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        batch_size = len(logits_text_per_image)
        logits_image_per_image =  image_embeds @ image_embeds.t()
        logits_text_per_text =  text_embeds @ text_embeds.t()
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * batch_size
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean()  * batch_size
        cyclic_loss =  inmodal_cyclic_loss + crossmodal_cyclic_loss

        return cyclic_loss



