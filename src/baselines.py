import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import DoubleEmbedding, GraphConvLayer, SpGraphAttentionLayer, GraphMultiHeadAttLayer, MultiHeadAttention


class GCN(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, embedding_dim, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout = dropout
        
        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gcnblocks = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout, act=False, transform=False) for i in range(self.layer)])
    
    def forward(self, **args):
        sr_embedding, tg_embedding = self.entity_embedding.weight
        for layer in self.gcnblocks:
            sr_embedding = layer(sr_embedding, self.adj_sr)
            tg_embedding = layer(tg_embedding, self.adj_tg)
        sr_embedding = F.normalize(sr_embedding, dim=1, p=2)
        tg_embedding = F.normalize(tg_embedding, dim=1, p=2)
        return sr_embedding, tg_embedding

class GAT(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, embedding_dim, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout = dropout

        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gatblocks = nn.ModuleList([SpGraphAttentionLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout, act=True, transform=True) for i in range(self.layer)])
    
    def forward(self, **args):
        sr_embedding, tg_embedding = self.entity_embedding.weight
        for layer in self.gatblocks:
            sr_embedding = layer(sr_embedding, self.adj_sr)
            tg_embedding = layer(tg_embedding, self.adj_tg)
        return sr_embedding, tg_embedding

class MTransE(nn.Module):
    def __init__(self, num_sr, num_tg, rel_num, embedding_dim, L1_flag=True) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.rel_num = rel_num
        self.embedding_dim = embedding_dim
        self.L1_flag = L1_flag

        self.sr_embedding = nn.Embedding(self.num_sr, self.embedding_dim)
        self.tg_embedding = nn.Embedding(self.num_tg, self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.rel_num, self.embedding_dim)
        self.transformation = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, **args):
        return self.sr_embedding.weight, self.tg_embedding.weight

    def km_loss(self, head_ids, rel_ids, tail_ids, type):
        '''Knowledge Model'''
        #print(len(head_ids), len(rel_ids), len(tail_ids))
        if type == "sr":
            head_embedding = self.sr_embedding.weight[head_ids]
            rel_embedding = self.rel_embedding.weight[rel_ids]
            tail_embedding = self.sr_embedding.weight[tail_ids]
        elif type == "tg":
            head_embedding = self.tg_embedding.weight[head_ids]
            rel_embedding = self.rel_embedding.weight[rel_ids]
            tail_embedding = self.tg_embedding.weight[tail_ids]
        if self.L1_flag:
            loss = torch.sum(torch.abs(head_embedding + rel_embedding - tail_embedding), 1)
            loss = torch.sum(loss)
        else:
            loss = torch.sum((head_embedding + rel_embedding - tail_embedding) ** 2, 1)
            loss = torch.sum(loss)
        return loss
    
    def am_loss(self, a1_align, a2_align):
        '''Alignment Model'''
        sr_embedding = self.sr_embedding.weight[a1_align]
        tg_embedding = self.tg_embedding.weight[a2_align]

        sr_embedding = self.transformation(sr_embedding)
        loss = torch.sum(torch.abs(sr_embedding - tg_embedding), 1)
        loss = torch.sum(loss)

        return loss

class GCN_Align(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, attr_num1, attr_num2, attr_weight_sr, attr_weight_tg, embedding_dim=1000, embedding_dim_attr=100, dropout=0.2, layer=2) -> None:
        super().__init__()
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.attr_num1 = attr_num1
        self.attr_num2 = attr_num2
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.layer = layer
        self.dropout = dropout
        self.embedding_dim = embedding_dim # for structure
        self.embedding_dim_attr = embedding_dim_attr # for attribute

        self.attr_weight_sr = attr_weight_sr # fixed
        self.attr_weight_tg = attr_weight_tg # fixed
        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=embedding_dim, init_type="xavier")
        self.gcnblocks_s = nn.ModuleList([GraphConvLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout) for i in range(self.layer)])
        self.gcnblocks_a_11 = GraphConvLayer(in_features=self.attr_num1, out_features=self.embedding_dim_attr)
        self.gcnblocks_a_12 = GraphConvLayer(in_features=self.attr_num2, out_features=self.embedding_dim_attr)
        self.gcnblocks_a_2 = GraphConvLayer(in_features=self.embedding_dim_attr, out_features=self.embedding_dim_attr)

    def forward(self, **args):
        sr_embedding_s, tg_embedding_s = self.entity_embedding.weight
        for layer in self.gcnblocks_s:
            sr_embedding_s = layer(sr_embedding_s, self.adj_sr)
            tg_embedding_s = layer(tg_embedding_s, self.adj_tg)
        sr_embedding_s = F.normalize(sr_embedding_s, dim=1, p=2)
        tg_embedding_s = F.normalize(tg_embedding_s, dim=1, p=2)
        
        sr_embedding_a = self.gcnblocks_a_11(self.attr_weight_sr, self.adj_sr)
        sr_embedding_a = self.gcnblocks_a_2(sr_embedding_a, self.adj_sr)
        tg_embedding_a = self.gcnblocks_a_12(self.attr_weight_tg, self.adj_tg)
        tg_embedding_a = self.gcnblocks_a_2(tg_embedding_a, self.adj_tg)

        # sr_embedding = torch.cat([sr_embedding_s, sr_embedding_a], dim=-1)
        # tg_embedding = torch.cat([tg_embedding_s, tg_embedding_a], dim=-1)

        return sr_embedding_s, tg_embedding_s, sr_embedding_a, tg_embedding_a


class GAEA(nn.Module):
    def __init__(self, num_sr, num_tg, adj_sr, adj_tg, rel_num, rel_adj_sr, rel_adj_tg, args) -> None:
        super().__init__()
        # --- parameters ---#
        self.num_sr = num_sr
        self.num_tg = num_tg
        self.adj_sr = adj_sr
        self.adj_tg = adj_tg
        self.rel_num = rel_num
        self.rel_adj_sr = rel_adj_sr
        self.rel_adj_tg = rel_adj_tg
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.n_head = args.n_head
        self.layer = args.layer
        self.direct = args.direct
        self.res = args.res

        # --- building blocks ---#
        self.dropout = nn.Dropout(args.dropout)
        self.entity_embedding = DoubleEmbedding(num_sr=self.num_sr, num_tg=self.num_tg, embedding_dim=self.ent_dim,
                                                init_type=args.init_type)
        self.relation_embedding = DoubleEmbedding(num_sr=self.rel_num, num_tg=self.rel_num, embedding_dim=self.rel_dim,
                                                  init_type=args.init_type)
        self.attentive_aggregators = nn.ModuleList([GraphMultiHeadAttLayer(in_features=self.ent_dim,
                                                                           out_features=self.ent_dim,
                                                                           n_head=self.n_head, dropout=args.dropout) for
                                                    i in range(self.layer)])
        self.multihead_attention = MultiHeadAttention(
            n_head=self.n_head,
            d_model=self.ent_dim,
            d_k=self.ent_dim,
            d_v=self.ent_dim,
            dropout=args.dropout
        )
        if self.direct:
            self.proj_head = nn.Sequential(
                nn.Linear(self.ent_dim * self.n_head + self.rel_dim * 2, self.ent_dim),
                nn.ReLU(),
                # nn.Linear(self.ent_dim, self.ent_dim),
                # nn.ReLU(),
            )
        else:
            self.proj_head = nn.Sequential(
                nn.Linear(self.ent_dim * self.n_head + self.rel_dim, self.ent_dim),
                nn.ReLU(),
                # nn.Linear(self.ent_dim, self.ent_dim),
                # nn.ReLU(),
            )

    def forward(self, aug_adj1=None, aug_rel_adj1=None, aug_adj2=None, aug_rel_adj2=None, phase="norm"):
        '''STEP: determine the KG information'''
        if phase in ["norm", "eval"]:
            adj_sr, adj_tg, rel_adj_sr, rel_adj_tg = self.adj_sr, self.adj_tg, self.rel_adj_sr, self.rel_adj_tg
        elif phase == "augment":
            adj_sr, adj_tg, rel_adj_sr, rel_adj_tg = aug_adj1, aug_adj2, aug_rel_adj1, aug_rel_adj2
        sr_embedding, tg_embedding = self.entity_embedding.weight
        sr_embedding, tg_embedding = self.dropout(sr_embedding), self.dropout(tg_embedding)
        sr_rel_embedding, tg_rel_embedding = self.relation_embedding.weight
        sr_rel_embedding, tg_rel_embedding = self.dropout(sr_rel_embedding), self.dropout(tg_rel_embedding)

        '''STEP: Relation Aggregator'''
        if self.direct:
            # for sr
            rel_adj_sr_in, rel_adj_sr_out = rel_adj_sr[0], rel_adj_sr[1]
            rel_rowsum_sr_in, rel_rowsum_sr_out = torch.sum(rel_adj_sr_in.to_dense(), dim=-1).unsqueeze(-1), torch.sum(
                rel_adj_sr_out.to_dense(), dim=-1).unsqueeze(-1)
            sr_rel_embedding_in = torch.mm(rel_adj_sr_in, sr_rel_embedding)
            sr_rel_embedding_in = sr_rel_embedding_in.div(rel_rowsum_sr_in + 1e-5)
            sr_rel_embedding_out = torch.mm(rel_adj_sr_out, sr_rel_embedding)
            sr_rel_embedding_out = sr_rel_embedding_out.div(rel_rowsum_sr_out + 1e-5)
            sr_rel_embedding = torch.cat([sr_rel_embedding_in, sr_rel_embedding_out], dim=-1)
            # for tg
            rel_adj_tg_in, rel_adj_tg_out = rel_adj_tg[0], rel_adj_tg[1]
            rel_rowsum_tg_in, rel_rowsum_tg_out = torch.sum(rel_adj_tg_in.to_dense(), dim=-1).unsqueeze(-1), torch.sum(
                rel_adj_tg_out.to_dense(), dim=-1).unsqueeze(-1)
            tg_rel_embedding_in = torch.mm(rel_adj_tg_in, tg_rel_embedding)
            tg_rel_embedding_in = tg_rel_embedding_in.div(rel_rowsum_tg_in + 1e-5)
            tg_rel_embedding_out = torch.mm(rel_adj_tg_out, tg_rel_embedding)
            tg_rel_embedding_out = tg_rel_embedding_out.div(rel_rowsum_tg_out + 1e-5)
            tg_rel_embedding = torch.cat([tg_rel_embedding_in, tg_rel_embedding_out], dim=-1)
        else:
            rel_rowsum_sr, rel_rowsum_tg = torch.sum(rel_adj_sr.to_dense(), dim=-1).unsqueeze(-1), torch.sum(
                rel_adj_tg.to_dense(), dim=-1).unsqueeze(-1)
            sr_rel_embedding = torch.mm(rel_adj_sr,
                                        sr_rel_embedding)  # [ent_num, rel_num] * [rel_num, rel_dim] => [ent_num, rel_dim]
            tg_rel_embedding = torch.mm(rel_adj_tg, tg_rel_embedding)
            sr_rel_embedding = sr_rel_embedding.div(rel_rowsum_sr)  # take mean value
            tg_rel_embedding = tg_rel_embedding.div(rel_rowsum_tg)

        '''STEP: Attentive Neighbor Aggregator'''
        sr_embedding_list, tg_embedding_list = list(), list()
        if self.res:
            sr_embedding_list.append(sr_embedding)
            tg_embedding_list.append(tg_embedding)
        for layer in self.attentive_aggregators:
            sr_embedding = layer(sr_embedding, adj_sr)
            tg_embedding = layer(tg_embedding, adj_tg)
            sr_embedding = F.normalize(sr_embedding, dim=1, p=2)
            tg_embedding = F.normalize(tg_embedding, dim=1, p=2)
            sr_embedding_list.append(self.dropout(sr_embedding))
            tg_embedding_list.append(self.dropout(tg_embedding))

        '''STEP: multi-range neighborhood fusion'''
        if self.res:  # apply residual link? default by "false".
            range = self.layer + 1
        else:
            range = self.layer
        sr_embedding = torch.cat(sr_embedding_list, dim=1).reshape(-1, range, self.ent_dim)  # [batch, range, ent_dim]
        sr_embedding, _ = self.multihead_attention(sr_embedding, sr_embedding,
                                                   sr_embedding)  # [batch, range, n_head * ent_dim]
        sr_embedding = torch.mean(sr_embedding, dim=1)  # [batch, n_head * ent_dim]
        sr_embedding = sr_embedding.squeeze(1)

        tg_embedding = torch.cat(tg_embedding_list, dim=1).reshape(-1, range, self.ent_dim)
        tg_embedding, _ = self.multihead_attention(tg_embedding, tg_embedding, tg_embedding)
        tg_embedding = torch.mean(tg_embedding, dim=1)
        tg_embedding = tg_embedding.squeeze(1)

        '''STEP: final fusion: neighbors + relation semantics'''
        sr_embedding = torch.cat([sr_embedding, sr_rel_embedding], dim=-1)
        tg_embedding = torch.cat([tg_embedding, tg_rel_embedding], dim=-1)

        return sr_embedding, tg_embedding

    def contrastive_loss(self, embeddings1, embeddings2, ent_num, sim_method="inner", t=0.08):
        '''calculate contrastive loss'''
        # step1: projection head for non-linear transformation
        embeddings1 = self.proj_head(embeddings1)
        embeddings2 = self.proj_head(embeddings2)

        # step2: symmetric loss function
        if sim_method == "cosine":
            embeddings1_abs = embeddings1.norm(dim=1)
            embeddings2_abs = embeddings2.norm(dim=1)
            logits = torch.einsum('ik,jk->ij', embeddings1, embeddings2) / (
                        torch.einsum('i,j->ij', embeddings1_abs, embeddings2_abs) + 1e-5)
            logits = logits / t
        elif sim_method == "inner":
            logits = torch.mm(embeddings1, embeddings2.T)
        labels = torch.arange(ent_num).to(embeddings1.device)
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.T, labels)
        return (loss_1 + loss_2) / 2