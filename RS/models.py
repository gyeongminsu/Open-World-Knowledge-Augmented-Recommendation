'''
-*- coding: utf-8 -*-
@File  : models.py
'''
import numpy as np
import sklearn.preprocessing
import torch
import torch.nn as nn
from layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention
from layers import Phi_function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import sklearn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class BaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()
        self.task = args.task
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len
        if self.task == 'rerank':
            self.max_list_len = dataset.max_list_len

        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.rnn_dp = args.rnn_dp
        self.output_dim = args.output_dim
        self.convert_dropout = args.convert_dropout
        self.convert_type = args.convert_type
        self.auxiliary_loss_weight = args.auxi_loss_weight

        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim
        self.dens_vec_num = 0

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim)
        self.attr_embedding = nn.Embedding(self.attr_num + 1, self.embed_dim)
        self.rating_embedding = nn.Embedding(self.rating_num + 1, self.embed_dim)
        if self.augment_num:
            self.convert_module = ConvertNet(args, self.dense_dim, self.convert_dropout, self.convert_type)
            self.dens_vec_num = args.convert_arch[-1] * self.augment_num

        self.module_inp_dim = self.get_input_dim()
        self.field_num = self.get_field_num()
        self.convert_loss = 0

    def process_input(self, inp):
        device = next(self.parameters()).device
        hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
        hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                 self.embed_dim * self.attr_fnum)
        hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                      self.embed_dim)
        hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
        hist_len = inp['hist_seq_len'].to(device)

        if self.task == 'ctr':
            iid_emb = self.item_embedding(inp['iid'].to(device))
            attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            # item_emb = item_emb.view(-1, self.itm_emb_dim)
            labels = inp['lb'].to(device)
            if self.augment_num:
                orig_dens_vec = [inp['hist_aug_vec'].to(device), inp['item_aug_vec'].to(device)]
                dens_vec = self.convert_module(orig_dens_vec)
            else:
                dens_vec, orig_dens_vec = None, None
            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_vec, labels
        elif self.task == 'rerank':
            iid_emb = self.item_embedding(inp['iid_list'].to(device))
            attr_emb = self.attr_embedding(inp['aid_list'].to(device)).view(-1, self.max_list_len,
                                                                            self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            item_emb = item_emb.view(-1, self.max_list_len, self.itm_emb_dim)
            labels = inp['lb_list'].to(device).view(-1, self.max_list_len)
            if self.augment_num:
                hist_aug = inp['hist_aug_vec'].to(device)
                item_list_aug = inp['item_aug_vec_list']
                orig_dens_list = [[hist_aug, item_aug.to(device)] for item_aug in item_list_aug]
                dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                dens_vec = torch.stack([dens for dens in dens_vec_list], dim=1)
            else:
                dens_vec, orig_dens_list = None, None

            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_list, labels
        else:
            raise NotImplementedError

    def get_input_dim(self):
        if self.task == 'ctr':
            return self.hist_emb_dim + self.itm_emb_dim + self.dens_vec_num
        elif self.task == 'rerank':
            return self.itm_emb_dim + self.dens_vec_num
        else:
            raise NotImplementedError

    def get_field_num(self):
        return self.item_fnum + self.augment_num + self.hist_fnum

    def get_filed_input(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels

    def process_rerank_inp(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            out = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            out = item_embedding
        return out, labels

    def get_ctr_output(self, logits, labels=None):
        outputs = {
            'logits': torch.sigmoid(logits),
            'labels': labels,
        }

        if labels is not None:
            if self.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.output_dim)), labels.float())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight

        return outputs

    def get_rerank_output(self, logits, labels=None, attn=False):
        outputs = {
            'logits': torch.sigmoid(logits),  # Ensure logits are constrained to [0, 1]
            'labels': labels,
        }

        if labels is not None:
            if attn:
                logits = attention_score(logits.view(-1, self.max_list_len), self.args.temperature)
                labels = attention_score(labels.float().view(-1, self.max_list_len), self.args.temperature)
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight
        return outputs

    def get_mask(self, length, max_len):
        device = next(self.parameters()).device
        rang = torch.arange(0, max_len).view(-1, max_len).to(device)
        batch_rang = rang.repeat([length.shape[0], 1])
        mask = batch_rang < torch.unsqueeze(length, dim=-1)
        return mask.unsqueeze(dim=-1).long()


class DeepInterestNet(BaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DeepInterestNet, self).__init__(args, dataset)

        self.map_layer = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
        # embedding of history item and candidate item should be the same
        self.attention_net = AttentionPoolingLayer(self.itm_emb_dim, self.dropout)

        # history embedding, item embedding, and user embedding
        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        mask = self.get_mask(hist_len, self.max_hist_len)

        user_behavior = self.map_layer(user_behavior)
        user_interest, _ = self.attention_net(query, user_behavior, mask)

        if self.augment_num:
            concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
        else:
            concat_input = torch.cat([user_interest, query], dim=-1)

        mlp_out = self.final_mlp(concat_input)
        logits = self.final_fc(mlp_out)
        out = self.get_ctr_output(logits, labels)
        return out


class DIEN(BaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DIEN, self).__init__(args, dataset)

        self.interest_extractor = InterestExtractor(self.hist_emb_dim, self.itm_emb_dim)
        self.interest_evolution = InterestEvolving(self.itm_emb_dim, gru_type=args.dien_gru, dropout=self.dropout)

        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, length, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        mask = self.get_mask(length, self.max_hist_len)
        length = torch.unsqueeze(length, dim=-1)
        masked_interest = self.interest_extractor(user_behavior, length)
        user_interest = self.interest_evolution(query, masked_interest, length, mask)  # [btz, hdsz]
        # user_interest = masked_interest.sum(dim=1)
        if self.augment_num:
            concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
        else:
            concat_input = torch.cat([user_interest, query], dim=-1)
        mlp_out = self.final_mlp(concat_input)
        logits = self.final_fc(mlp_out)
        out = self.get_ctr_output(logits, labels)
        return out


class DCN(BaseModel):
    '''
    DCNv1
    '''
    def __init__(self, args, mode, dataset):
        super(DCN, self).__init__(args, dataset)
        self.deep_arch = args.dcn_deep_arch
        self.cross_net = CrossNet(self.module_inp_dim, args.dcn_cross_num, mode)
        self.deep_net = MLP(self.deep_arch, self.module_inp_dim, self.dropout)
        final_inp_dim = self.module_inp_dim + self.deep_arch[-1]
        self.final_mlp = MLP(self.final_mlp_arch, final_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)


    def forward(self, inp):
        '''
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        '''
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)

        deep_part = self.deep_net(inp)
        cross_part = self.cross_net(inp)

        final_inp = torch.cat([deep_part, cross_part], dim=1)
        mlp_out = self.final_mlp(final_inp)
        logits = self.final_fc(mlp_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class DeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(DeepFM, self).__init__(args, dataset)
        # FM
        self.fm_first_iid_emb = nn.Embedding(self.item_num + 1, 1)
        self.fm_first_aid_emb = nn.Embedding(self.attr_num + 1, 1)
        self.fm_first_dense_weight = nn.Parameter(torch.rand([self.dens_vec_num, 1]))
        # DNN
        self.deep_part = MLP(args.deepfm_deep_arch, self.module_inp_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.deepfm_deep_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            dnn_inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

        device = next(self.parameters()).device
        # fm first order
        iid_first = self.fm_first_iid_emb(inp['iid'].to(device)).view(-1, 1)
        aid_first = self.fm_first_aid_emb(inp['aid'].to(device)).view(-1, self.attr_fnum)
        linear_sparse_logit = torch.sum(torch.cat([iid_first, aid_first], dim=1), dim=1).view(-1, 1)
        if self.augment_num:
            linear_dense_logit = dens_vec.matmul(self.fm_first_dense_weight).view(-1, 1)
            fm_logit = linear_sparse_logit + linear_dense_logit
        else:
            fm_logit = linear_sparse_logit

        # fm second order
        fm_second_inp = torch.cat([item_embedding, user_behavior], dim=1)
        fm_second_inp = fm_second_inp.view(-1, self.item_fnum + self.hist_fnum, self.embed_dim)

        square_of_sum = torch.pow(torch.sum(fm_second_inp, dim=1, keepdim=True), 2)  # shape: (batch_size,1,embedding_size)
        sum_of_square = torch.sum(torch.pow(fm_second_inp, 2), dim=1, keepdim=True)  # shape: (batch_size,1,embedding_size)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # shape: (batch_size,1)
        # print(cross_term.shape)
        fm_logit += cross_term

        # dnn
        deep_out = self.deep_part(dnn_inp)

        logits = fm_logit + self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class xDeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(xDeepFM, self).__init__(args, dataset)
        input_dim = self.field_num * args.embed_dim
        cin_layer_units = args.cin_layer_units
        self.cin = CIN(self.field_num, cin_layer_units)
        self.dnn = MLP(args.final_mlp_arch, input_dim, self.dropout)
        final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
        self.final_fc = nn.Linear(final_dim, args.output_dim)

    def forward(self, inp):
        inp, labels = self.get_filed_input(inp)

        final_vec = self.cin(inp)
        dnn_vec = self.dnn(inp.flatten(start_dim=1))
        final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        logits = self.final_fc(final_vec)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class AutoInt(BaseModel):
    def __init__(self, args, dataset):
        super(AutoInt, self).__init__(args, dataset)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(args.embed_dim if i == 0 else args.num_attn_heads * args.attn_size,
                                     attention_dim=args.attn_size,
                                     num_heads=args.num_attn_heads,
                                     dropout_rate=args.dropout,
                                     use_residual=args.res_conn,
                                     use_scale=args.attn_scale,
                                     layer_norm=False,
                                     align_to='output')
              for i in range(args.num_attn_layers)])
        final_dim = self.field_num * args.attn_size * args.num_attn_heads

        self.attn_out = nn.Linear(final_dim, 1)

    def forward(self, inp):
        inp, labels = self.get_filed_input(inp)
        attention_out = self.self_attention(inp)
        attention_out = torch.flatten(attention_out, start_dim=1)

        logits = self.attn_out(attention_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class FiBiNet(BaseModel):
    def __init__(self, args, dataset):
        super(FiBiNet, self).__init__(args, dataset)
        self.senet_layer = SqueezeExtractionLayer(self.field_num, args.reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(self.embed_dim, self.field_num, args.bilinear_type)
        final_dim = self.field_num * (self.field_num - 1) * self.embed_dim
        self.dnn = MLP(args.final_mlp_arch, final_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        feat_embed, labels = self.get_filed_input(inp)
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)

        logits = self.fc_out(self.dnn(comb_out))
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class FiGNN(BaseModel):
    def __init__(self, args, dataset):
        super(FiGNN, self).__init__(args, dataset)
        self.fignn = FiGNNBlock(self.field_num, self.embed_dim, args.gnn_layer_num,
                                args.res_conn, args.reuse_graph_layer)
        self.fc = AttentionalPrediction(self.field_num, self.embed_dim)

    def forward(self, inp):
        feat_embed, labels = self.get_filed_input(inp)
        h_out = self.fignn(feat_embed)
        logits = self.fc(h_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class DLCM(BaseModel):
    def __init__(self, args, dataset):
        super(DLCM, self).__init__(args, dataset)
        self.gru = torch.nn.GRU(self.module_inp_dim, self.hidden_size, dropout=self.rnn_dp, batch_first=True)
        self.phi_function = Phi_function(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        seq_state, final_state = self.gru(processed_inp)
        final_state = torch.squeeze(final_state, dim=0)

        scores = self.phi_function(seq_state, final_state)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class PRM(BaseModel):
    def __init__(self, args, dataset):
        super(PRM, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.pos_embedding = torch.tensor(self.get_pos_embedding(self.max_list_len,
                                                                 self.module_inp_dim)).float().to(args.device)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def get_pos_embedding(self, max_len, d_emb):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        item_embed = processed_inp + self.pos_embedding

        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class SetRank(BaseModel):
    def __init__(self, args, dataset):
        super(SetRank, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels, attn=True)
        return outputs


class MIR(BaseModel):
    def __init__(self, args, dataset):
        super(MIR, self).__init__(args, dataset)
        self.intra_item_attn = nn.MultiheadAttention(self.itm_emb_dim, args.n_head, batch_first=True,
                                                     dropout=args.attn_dp)
        self.intra_hist_gru = nn.GRU(self.hist_emb_dim, self.hidden_size, dropout=self.rnn_dp,
                                     batch_first=True)
        self.i_fnum = self.item_fnum * 2
        self.h_fnum = self.hist_fnum + self.hidden_size // self.embed_dim

        self.set2list_attn = SLAttention(self.i_fnum, self.h_fnum, self.embed_dim,
                                         self.max_list_len, self.max_hist_len)
        self.module_inp_dim = (self.item_fnum * 3 + self.h_fnum * 2) * self.embed_dim
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        cross_item, _ = self.intra_item_attn(item_embedding, item_embedding, item_embedding)
        cross_hist, _ = self.intra_hist_gru(user_behavior)
        user_seq = torch.cat([user_behavior, cross_hist], dim=-1)
        hist_mean = torch.mean(user_seq, dim=1, keepdim=True)
        hist_mean = hist_mean.repeat([1, self.max_list_len, 1])
        cat_item = torch.cat([item_embedding, cross_item], dim=-1)

        v, q, _, _ = self.set2list_attn(cat_item, user_seq)
        mlp_inp = torch.cat([v, q, item_embedding, hist_mean], dim=-1)
        mlp_out = self.mlp(mlp_inp)
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs
    
class LightGCN(BaseModel):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__(args, dataset)
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.dropout
        
        # Initialize embeddings
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            'item_emb': nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })

        # Get normalized adj matrix
        self.register_buffer('norm_adj_matrix', self.get_norm_adj_matrix())

    def get_norm_adj_matrix(self):
        # Create adjacency matrix from user-item interactions
        adj_mat = torch.zeros((self.max_hist_len + self.max_list_len, 
                             self.max_hist_len + self.max_list_len))
        
        # Fill adjacency matrix based on interactions
        adj_mat[:self.max_hist_len, self.max_hist_len:] = 1
        adj_mat[self.max_hist_len:, :self.max_hist_len] = 1
        
        adj_mat = adj_mat + torch.eye(self.max_hist_len + self.max_list_len)

        # Normalize adjacency matrix
        degree = torch.sum(adj_mat, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        
        
        # D^(-1/2) * A * D^(-1/2)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(degree_inv_sqrt, adj_mat), degree_inv_sqrt)
        
        return norm_adj

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        batch_size = item_embedding.size(0)

        # Light Graph Convolution
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                  self.embedding_dict['item_emb']], dim=0)
        embeddings_list = [all_embeddings]
        
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        # Layer combination
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, 
            [self.max_hist_len, self.max_list_len], dim=0)
        
        # Reshape for batch processing
        user_embeddings = user_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, max_hist_len, embed_dim]
        item_embeddings = item_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, max_list_len, embed_dim]

        # Calculate scores
        scores = torch.matmul(user_all_embeddings, item_all_embeddings.t())
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        
        # Calculate scores for each batch
        scores = torch.bmm(user_embeddings, item_embeddings.transpose(1, 2))  # [batch_size, max_hist_len, max_list_len]
        scores = torch.mean(scores, dim=1)  # [batch_size, max_list_len]
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedLightGCN(BaseModel):
    def __init__(self, args, dataset):
        super(EnhancedLightGCN, self).__init__(args, dataset)
        self.n_layers = 3  # Optimal number of layers
        self.keep_prob = 1 - args.dropout
        self.alpha = 0.5  # Layer weight parameter
        self.hard_neg_margin = 0.3
        self.curriculum_temp = 1.0  # For curriculum learning
        
        # Initialize embeddings with Xavier initialization
        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            "item_emb": nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })
        
        nn.init.xavier_uniform_(self.embedding_dict["user_emb"])
        nn.init.xavier_uniform_(self.embedding_dict["item_emb"])
        
        # Layer-specific transformation
        self.layer_transforms = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.n_layers)
        ])
        
        # Get normalized adj matrix with self-connections
        self.register_buffer("norm_adj_matrix", self.get_norm_adj_matrix())
        
    def get_norm_adj_matrix(self):
        adj_mat = torch.zeros((self.max_hist_len + self.max_list_len,
                             self.max_hist_len + self.max_list_len))
        
        # Enhanced adjacency matrix with self-connections
        adj_mat[:self.max_hist_len, self.max_hist_len:] = 1
        adj_mat[self.max_hist_len:, :self.max_hist_len] = 1
        adj_mat = adj_mat + torch.eye(self.max_hist_len + self.max_list_len)
        
        # Symmetric normalization
        degree = torch.sum(adj_mat, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(degree_inv_sqrt, adj_mat), degree_inv_sqrt)
        
        return norm_adj
    
    def mix_propagate(self, all_embeddings, current_layer):
        # MixGCF propagation
        neighbors_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
        transformed_embeddings = self.layer_transforms[current_layer](neighbors_embeddings)
        mixed_embeddings = self.alpha * transformed_embeddings + (1 - self.alpha) * all_embeddings
        return F.normalize(mixed_embeddings, p=2, dim=1)
    
    def get_hard_negatives(self, user_embeddings, item_embeddings):
        # Calculate similarity scores
        similarity = torch.matmul(user_embeddings, item_embeddings.transpose(1, 2))
        
        # Find hard negatives (items with high similarity but are not positive)
        hard_negative_scores, hard_negative_indices = torch.topk(similarity, k=5, dim=2)
        return hard_negative_indices
    
    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        batch_size = item_embedding.size(0)
        
        # Initial embeddings
        all_embeddings = torch.cat([self.embedding_dict["user_emb"], 
                                  self.embedding_dict["item_emb"]], dim=0)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        # Multi-layer propagation with MixGCF
        embeddings_list = [all_embeddings]
        for layer in range(self.n_layers):
            all_embeddings = self.mix_propagate(all_embeddings, layer)
            embeddings_list.append(all_embeddings)
        
        # Weighted sum of layer embeddings
        weights = torch.softmax(torch.tensor([1.0] * (self.n_layers + 1)), dim=0)
        all_embeddings = sum([emb * w for emb, w in zip(embeddings_list, weights)])
        
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.max_hist_len, self.max_list_len], dim=0
        )
        
        # Reshape for batch processing
        user_embeddings = user_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        item_embeddings = item_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Get hard negatives
        hard_neg_indices = self.get_hard_negatives(user_embeddings, item_embeddings)
        
        # Calculate attention scores with temperature scaling
        attention_logits = torch.bmm(user_embeddings, item_embeddings.transpose(1, 2))
        attention_weights = F.softmax(attention_logits / self.curriculum_temp, dim=1)
        
        # Combine attention scores with hard negative information
        hard_negative_mask = torch.zeros_like(attention_weights).scatter_(
            2, hard_neg_indices, self.hard_neg_margin
        )
        attention_weights = attention_weights - hard_negative_mask
        
        # Final scoring
        scores = torch.sum(attention_weights, dim=1)
        scores = torch.sigmoid(scores)
        
        # Update curriculum temperature
        self.curriculum_temp = max(0.5, self.curriculum_temp * 0.995)  # Gradually decrease temperature
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class LLM4Rerank(BaseModel):
    def __init__(self, args, dataset):
        super(LLM4Rerank, self).__init__(args, dataset)
        
        # LLM-specific parameters
        self.llm_heads = args.llm_heads 
        self.llm_layers = args.llm_layers
        self.llm_ff_dim = args.llm_ff_dim
        
        # Input dimension
        self.inp_dim = self.itm_emb_dim
        if self.augment_num:
            self.inp_dim += self.dens_vec_num  # Ensure this matches the actual embedding dimension
        
        # Debug print
        #print(f"self.inp_dim: {self.inp_dim}")
        #print(f"self.itm_emb_dim: {self.itm_emb_dim}")
        #print(f"self.dens_vec_num: {self.dens_vec_num}")
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.inp_dim,
            num_heads=self.llm_heads,
            batch_first=True,
            dropout=args.dropout
        )
        
        # Positional embedding layer
        self.max_seq_len = self.max_list_len
        self.pos_embedding_layer = nn.Embedding(self.max_seq_len, self.inp_dim)
        
        # MLP layers
        self.mlp = MLP(
            args.final_mlp_arch,
            self.inp_dim * 2,
            args.dropout
        )
        
        self.score_proj = nn.Linear(args.final_mlp_arch[-1], 1)
        
    def forward(self, inp):
        # Process input
        item_embedding, _, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        
        # Concatenate dens_vec if present
        if dens_vec is not None:
            item_embedding = torch.cat([item_embedding, dens_vec], dim=-1)
        
        # Debug print
        #print(f"item_embedding shape: {item_embedding.shape}")
        
        # Sequence length
        seq_len = item_embedding.size(1)
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=item_embedding.device).unsqueeze(0).expand(item_embedding.size(0), seq_len)
        
        # Positional embeddings
        pos_embed = self.pos_embedding_layer(position_ids)
        
        # Debug print
        #print(f"pos_embed shape: {pos_embed.shape}")
        
        # Add positional embeddings
        item_embed_pos = item_embedding + pos_embed
        
        # Self attention
        attn_out, _ = self.attention(
            item_embed_pos,
            item_embed_pos,
            item_embed_pos
        )
        
        # Concatenate attention output with original embeddings
        mlp_input = torch.cat([attn_out, item_embed_pos], dim=-1)
        
        # MLP scoring
        mlp_out = self.mlp(mlp_input)
        scores = torch.sigmoid(self.score_proj(mlp_out).squeeze(-1))
        
        # Get final scores
        outputs = self.get_rerank_output(scores, labels, attn=True)
        
        return outputs
       
class EnhancedLLM4Rerank(BaseModel):
    def __init__(self, args, dataset):
        super(EnhancedLLM4Rerank, self).__init__(args, dataset)
        
        # LLM-specific parameters
        self.llm_heads = args.llm_heads 
        self.llm_layers = args.llm_layers
        self.llm_ff_dim = args.llm_ff_dim
        
        # Input dimension
        self.inp_dim = self.itm_emb_dim
        if self.augment_num:
            self.inp_dim += self.dens_vec_num
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(self.inp_dim)
        self.layer_norm2 = nn.LayerNorm(self.inp_dim)
        
        # Multi-layer transformer blocks
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.inp_dim,
                    num_heads=self.llm_heads,
                    batch_first=True,
                    dropout=args.dropout
                ),
                'ff_layer': nn.Sequential(
                    nn.Linear(self.inp_dim, self.llm_ff_dim),
                    nn.GELU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(self.llm_ff_dim, self.inp_dim),
                    nn.Dropout(args.dropout)
                )
            }) for _ in range(self.llm_layers)
        ])
        
        # Positional embedding
        self.max_seq_len = self.max_list_len
        self.pos_embedding_layer = nn.Embedding(self.max_seq_len, self.inp_dim)
        
        # Cross-attention layer for global context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.inp_dim,
            num_heads=self.llm_heads,
            batch_first=True,
            dropout=args.dropout
        )
        
        # Final MLP with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(self.inp_dim * 2, args.final_mlp_arch[0]),
            nn.LayerNorm(args.final_mlp_arch[0]),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.final_mlp_arch[0], args.final_mlp_arch[1]),
            nn.LayerNorm(args.final_mlp_arch[1]),
            nn.GELU(),
            nn.Dropout(args.dropout)
        )
        
        self.score_proj = nn.Linear(args.final_mlp_arch[-1], 1)
        
    def forward(self, inp):
        item_embedding, _, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        
        if dens_vec is not None:
            item_embedding = torch.cat([item_embedding, dens_vec], dim=-1)
        
        # Add positional embeddings
        seq_len = item_embedding.size(1)
        position_ids = torch.arange(seq_len, device=item_embedding.device).unsqueeze(0).expand(item_embedding.size(0), seq_len)
        pos_embed = self.pos_embedding_layer(position_ids)
        x = item_embedding + pos_embed
        
        # Apply transformer layers with residual connections
        for layer in self.transformer_layers:
            # Self-attention block
            residual = x
            x = self.layer_norm1(x)
            attn_out, _ = layer['attention'](x, x, x)
            x = residual + attn_out
            
            # Feed-forward block
            residual = x
            x = self.layer_norm2(x)
            x = residual + layer['ff_layer'](x)
        
        # Global context with cross-attention
        global_context, _ = self.cross_attention(
            x,
            x.mean(dim=1, keepdim=True).expand(-1, seq_len, -1),
            x
        )
        
        # Combine local and global features
        combined_features = torch.cat([x, global_context], dim=-1)
        
        # Final MLP and scoring
        mlp_out = self.mlp(combined_features)
        scores = torch.sigmoid(self.score_proj(mlp_out).squeeze(-1))
        
        outputs = self.get_rerank_output(scores, labels, attn=True)
        return outputs
       
class LLM4Rerank_1(BaseModel):
    def __init__(self, args, dataset):
        super(LLM4Rerank_1, self).__init__(args, dataset)
        
        self.llm_heads = args.llm_heads 
        self.llm_layers = args.llm_layers
        self.llm_ff_dim = args.llm_ff_dim
        
        # Input dimension
        self.inp_dim = self.itm_emb_dim
        if self.augment_num:
            self.inp_dim += self.dens_vec_num
            
        # Layer normalization
        self.input_norm = nn.LayerNorm(self.inp_dim)
        
        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.inp_dim,
            nhead=4,
            dim_feedforward=self.inp_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )
        
        # Positional embedding
        self.max_seq_len = self.max_list_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.inp_dim))
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.inp_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Initialize positional embeddings
        nn.init.xavier_uniform_(self.pos_embedding)
        
    def forward(self, inp):
        item_embedding, _, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        
        if dens_vec is not None:
            item_embedding = torch.cat([item_embedding, dens_vec], dim=-1)
        
        # Add positional embeddings
        x = item_embedding + self.pos_embedding[:, :item_embedding.size(1), :]
        
        # Layer normalization
        x = self.input_norm(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Output scores
        scores = torch.sigmoid(self.output_layer(x).squeeze(-1))
        
        # Numerical stability
        scores = torch.clamp(scores, min=1e-7, max=1-1e-7)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class LightGCNPlusPlus(BaseModel):
    def __init__(self, args, dataset):
        super(LightGCNPlusPlus, self).__init__(args, dataset)
        self.n_layers = args.n_layers
        self.keep_prob = 1 - args.dropout
        
        # Initialize embeddings
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            'item_emb': nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })
        
        # New parameters for LightGCN++
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        self.norm_scale = nn.Parameter(torch.ones(1))
        self.neighbor_weights = nn.Parameter(torch.ones(self.n_layers))
        
        self.register_buffer('norm_adj_matrix', self.get_norm_adj_matrix())

    def get_norm_adj_matrix(self):
        adj_mat = torch.zeros((self.max_hist_len + self.max_list_len, 
                             self.max_hist_len + self.max_list_len))
        
        adj_mat[:self.max_hist_len, self.max_hist_len:] = 1
        adj_mat[self.max_hist_len:, :self.max_hist_len] = 1
        
        adj_mat = adj_mat + torch.eye(self.max_hist_len + self.max_list_len)

        degree = torch.sum(adj_mat, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(degree_inv_sqrt, adj_mat), degree_inv_sqrt)
        
        return norm_adj

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        batch_size = item_embedding.size(0)

        # Enhanced Light Graph Convolution
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                  self.embedding_dict['item_emb']], dim=0)
        embeddings_list = [all_embeddings]
        
        # Flexible norm scaling and neighbor weighting
        current_embedding = all_embeddings * self.norm_scale
        
        for layer in range(self.n_layers):
            neighbor_embedding = torch.sparse.mm(self.norm_adj_matrix, current_embedding)
            current_embedding = neighbor_embedding * self.neighbor_weights[layer]
            embeddings_list.append(current_embedding)
            
        # Improved layer combination with learnable weights
        all_embeddings = torch.stack(embeddings_list, dim=1)
        weights = F.softmax(self.layer_weights, dim=0)
        all_embeddings = torch.sum(all_embeddings * weights.view(1, -1, 1), dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, 
            [self.max_hist_len, self.max_list_len], dim=0)
        
        user_embeddings = user_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        item_embeddings = item_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        # Calculate scores with enhanced embeddings
        scores = torch.bmm(user_embeddings, item_embeddings.transpose(1, 2))
        scores = torch.mean(scores, dim=1)
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class NGCF(BaseModel):
    def __init__(self, args, dataset):
        super(NGCF, self).__init__(args, dataset)
        self.n_layers = args.n_layers
        
        # Initialize embeddings
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            'item_emb': nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })
        
        # NGCF layers
        self.layers = nn.ModuleList()
        self.dropout = args.dropout
        
        self.weight_size_list = [self.embed_dim] + [self.embed_dim] * self.n_layers
        
        for k in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                'W1': nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1], bias=True),
                'W2': nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1], bias=True)
            }))
        
        # Get normalized adjacency matrix
        self.register_buffer('norm_adj_matrix', self.get_norm_adj_matrix())
        
    def get_norm_adj_matrix(self):
        adj_mat = torch.zeros((self.max_hist_len + self.max_list_len, 
                             self.max_hist_len + self.max_list_len))
        
        # Fill adjacency matrix based on user-item interactions
        adj_mat[:self.max_hist_len, self.max_hist_len:] = 1
        adj_mat[self.max_hist_len:, :self.max_hist_len] = 1
        adj_mat = adj_mat + torch.eye(self.max_hist_len + self.max_list_len)
        
        # Normalize adjacency matrix
        degree = torch.sum(adj_mat, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(degree_inv_sqrt, adj_mat), degree_inv_sqrt)
        
        return norm_adj
        
    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        batch_size = item_embedding.size(0)
        
        # Initial embeddings
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                  self.embedding_dict['item_emb']], dim=0)
        embeddings_list = [all_embeddings]
        
        # Graph Convolution layers
        for layer in self.layers:
            # Message construction
            side_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            
            # Message passing
            sum_embeddings = F.leaky_relu(layer['W1'](side_embeddings))
            bi_embeddings = F.leaky_relu(layer['W2'](torch.mul(all_embeddings, side_embeddings)))
            
            # Aggregation
            all_embeddings = sum_embeddings + bi_embeddings
            
            # Dropout
            if self.training:
                all_embeddings = F.dropout(all_embeddings, self.dropout)
            
            # L2 normalize
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)
            
        # Layer combination
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, 
            [self.max_hist_len, self.max_list_len], dim=0)
        
        # Reshape for batch processing
        user_embeddings = user_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        item_embeddings = item_all_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Calculate scores
        scores = torch.bmm(user_embeddings, item_embeddings.transpose(1, 2))
        scores = torch.mean(scores, dim=1)
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class BasicDLCM(BaseModel):
    def __init__(self, args, dataset):
        super(BasicDLCM, self).__init__(args, dataset)
        # Basic feature dimension without augmented vectors
        self.basic_inp_dim = self.itm_emb_dim
        
        self.gru = torch.nn.GRU(self.basic_inp_dim, self.hidden_size, dropout=self.rnn_dp, batch_first=True)
        self.phi_function = Phi_function(self.hidden_size, self.hidden_size, self.dropout)
        
    def forward(self, inp):
        # Get only item embeddings without augmented vectors
        item_embedding, user_behavior, hist_len, _, _, labels = self.process_input(inp)
        seq_state, final_state = self.gru(item_embedding)
        final_state = torch.squeeze(final_state, dim=0)
        
        scores = self.phi_function(seq_state, final_state)
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class BasicLLM4Rerank(BaseModel):
    def __init__(self, args, dataset):
        super(BasicLLM4Rerank, self).__init__(args, dataset)
        self.basic_inp_dim = self.itm_emb_dim
            
        self.input_norm = nn.LayerNorm(self.basic_inp_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.basic_inp_dim,
            nhead=4,
            dim_feedforward=self.basic_inp_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_list_len, self.basic_inp_dim))
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.basic_inp_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        nn.init.xavier_uniform_(self.pos_embedding)
        
    def forward(self, inp):
        item_embedding, _, hist_len, _, _, labels = self.process_input(inp)
        
        # Add positional embeddings
        x = item_embedding + self.pos_embedding[:, :item_embedding.size(1), :]
        x = self.input_norm(x)
        x = self.transformer_encoder(x)
        scores = torch.sigmoid(self.output_layer(x).squeeze(-1))
        scores = torch.clamp(scores, min=1e-7, max=1-1e-7)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class BasicLightGCN(BaseModel):
    def __init__(self, args, dataset):
        super(BasicLightGCN, self).__init__(args, dataset)
        self.n_layers = args.n_layers
        
        # Only basic embeddings without augmented vectors
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            'item_emb': nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })
        
        self.register_buffer('norm_adj_matrix', self.get_norm_adj_matrix())
        
    def get_norm_adj_matrix(self):
        n_users = self.max_hist_len
        n_items = self.max_list_len
    
        # 사용자-아이템 인접 행렬 생성
        adj_mat = torch.zeros((n_users + n_items, n_users + n_items))
        adj_mat[:n_users, n_users:] = 1  # 사용자-아이템 연결
        adj_mat[n_users:, :n_users] = 1  # 아이템-사용자 연결 (전치)
    
        # 정규화된 라플라시안 행렬 계산
        rowsum = adj_mat.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_mat), d_mat_inv_sqrt)
    
        # 희소 텐서로 변환
        indices = torch.nonzero(norm_adj).t()
        values = norm_adj[indices[0], indices[1]]
    
        return torch.sparse_coo_tensor(
            indices, values, 
            size=(n_users + n_items, n_users + n_items)
        )
    
    def forward(self, inp):
        item_embedding, user_behavior, hist_len, _, _, labels = self.process_input(inp)
        batch_size = item_embedding.size(0)
        
        # Basic GCN propagation
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                  self.embedding_dict['item_emb']], dim=0)
        embeddings_list = [all_embeddings]
        
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        all_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)
        user_emb, item_emb = torch.split(all_embeddings, [self.max_hist_len, self.max_list_len])
        
        user_emb = user_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        item_emb = item_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        
        scores = torch.bmm(user_emb, item_emb.transpose(1, 2))
        scores = torch.mean(scores, dim=1)
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs

class BasicNGCF(BaseModel):
    def __init__(self, args, dataset):
        super(BasicNGCF, self).__init__(args, dataset)
        self.n_layers = args.n_layers
        
        # Basic embeddings
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.randn(self.max_hist_len, self.embed_dim)),
            'item_emb': nn.Parameter(torch.randn(self.max_list_len, self.embed_dim))
        })
        
        # NGCF layers without augmented features
        self.layers = nn.ModuleList()
        self.weight_size_list = [self.embed_dim] * (self.n_layers + 1)
        
        for k in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                'W1': nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1], bias=True),
                'W2': nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1], bias=True)
            }))
        
        self.register_buffer('norm_adj_matrix', self.get_norm_adj_matrix())
    
    def get_norm_adj_matrix(self):
        n_users = self.max_hist_len
        n_items = self.max_list_len
    
        # 사용자-아이템 인접 행렬 생성
        adj_mat = torch.zeros((n_users + n_items, n_users + n_items))
        adj_mat[:n_users, n_users:] = 1  # 사용자-아이템 연결
        adj_mat[n_users:, :n_users] = 1  # 아이템-사용자 연결 (전치)
    
        # 정규화된 라플라시안 행렬 계산
        rowsum = adj_mat.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_mat), d_mat_inv_sqrt)
    
        # 희소 텐서로 변환
        indices = torch.nonzero(norm_adj).t()
        values = norm_adj[indices[0], indices[1]]
    
        return torch.sparse_coo_tensor(
            indices, values, 
            size=(n_users + n_items, n_users + n_items)
        )
    
    def forward(self, inp):
        item_embedding, user_behavior, hist_len, _, _, labels = self.process_input(inp)
        batch_size = item_embedding.size(0)
        
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], 
                                  self.embedding_dict['item_emb']], dim=0)
        embeddings_list = [all_embeddings]
        
        for layer in self.layers:
            # Message passing without augmented features
            side_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            sum_embeddings = F.leaky_relu(layer['W1'](side_embeddings))
            bi_embeddings = F.leaky_relu(layer['W2'](torch.mul(all_embeddings, side_embeddings)))
            all_embeddings = sum_embeddings + bi_embeddings
            
            if self.training:
                all_embeddings = F.dropout(all_embeddings, self.dropout)
            
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)
            
        all_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)
        user_emb, item_emb = torch.split(all_embeddings, [self.max_hist_len, self.max_list_len])
        
        user_emb = user_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        item_emb = item_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        
        scores = torch.bmm(user_emb, item_emb.transpose(1, 2))
        scores = torch.mean(scores, dim=1)
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs