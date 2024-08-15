#!/usr/bin/env python3
import torch
import time
import math
import ParamSpCONV as SpCONV
from PCSR import *
from graph_input import *

n_heads = 1
n_output = 8


class GCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, weight, ParamSpMM):
        ctx.save_for_backward(H, weight)
        ctx.ParamSpMM = ParamSpMM
        # H_prime = ParamSpMM.forward(H, weight)
        if ParamSpMM.if_split == False:
            if ParamSpMM.vec_size == 1:
                H_prime = SpCONV.csr_spmm_forward(
                    H, weight, ParamSpMM.rowPtr, ParamSpMM.colIdx, ParamSpMM.val, ParamSpMM.CF, ParamSpMM.blkWarpNum
                )
            else:
                H_prime = SpCONV.bcsr_spmm_forward(
                    H, weight, ParamSpMM.rowPtr, ParamSpMM.colIdx, ParamSpMM.val, ParamSpMM.CF, ParamSpMM.blkWarpNum
                )
        else:
            if ParamSpMM.vec_size == 1:
                H_prime = SpCONV.csr_split_spmm_forward(
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.target_row,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )
            else:
                H_prime = SpCONV.bcsr_split_spmm_forward(
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.target_row,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )
        return H_prime

    @staticmethod
    def backward(ctx, d_output):
        H, weight = ctx.saved_tensors
        ParamSpMM = ctx.ParamSpMM
        # d_input, d_weight = ParamSpMM.backward(d_output, H, weight)
        if ParamSpMM.if_split == False:
            if ParamSpMM.vec_size == 1:
                d_input, d_weight = SpCONV.csr_spmm_backward(
                    d_output,
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )
            else:
                d_input, d_weight = SpCONV.bcsr_spmm_backward(
                    d_output,
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )
        else:
            if ParamSpMM.vec_size == 1:
                d_input, d_weight = SpCONV.csr_split_spmm_backward(
                    d_output,
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.target_row,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )
            else:
                d_input, d_weight = SpCONV.bcsr_split_spmm_backward(
                    d_output,
                    H,
                    weight,
                    ParamSpMM.rowPtr,
                    ParamSpMM.colIdx,
                    ParamSpMM.val,
                    ParamSpMM.target_row,
                    ParamSpMM.CF,
                    ParamSpMM.blkWarpNum,
                )

        return d_input, d_weight, None


class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, H, spmm_dict):
        if self.weights.size(1) not in spmm_dict:
            raise ValueError("SpMMDecider not support dim: " + str(self.weights.size(1)))
        else:
            ParamSpMM = spmm_dict[self.weights.size(1)]
        return GCNFunction.apply(H, self.weights, ParamSpMM)


class AGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, attention_w, ParamSpMM, colIdx):
        # GEMM node update
        X_prime = torch.mm(X, weights)
        edge_feature = ParamSpMM.AGNN_sddmm(X_prime)

        # Edge Attention Generation: [n_e, n_head]
        if edge_feature.dim() == 1:
            edge_feature = edge_feature.unsqueeze(-1)
            edge_attentions = (edge_feature * attention_w).squeeze()
        else:
            edge_attentions = torch.zeros_like(edge_feature)
            edge_attentions[:, 0] = edge_feature[:, 0] * attention_w
            edge_attentions[:, 1] = edge_feature[:, 1] * attention_w

        # SpMM_AGNN: Neighbor AggreAGNNion.
        X_prime = ParamSpMM.AGNN_spmm(edge_attentions, X_prime)

        ctx.save_for_backward(X, weights, edge_attentions)
        ctx.colIdx = colIdx
        ctx.ParamSpMM = ParamSpMM
        # print("==========After Aggreation=========")
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        ParamSpMM = ctx.ParamSpMM
        colIdx = ctx.colIdx
        X, weights, edge_attentions = ctx.saved_tensors
        d_input_prime = ParamSpMM.AGNN_spmm(edge_attentions, d_output)

        d_input = torch.mm(d_input_prime, weights.t().contiguous())
        d_weights = torch.mm(X.t().contiguous(), d_input_prime)

        # attention weight back propaAGNNion.
        d_attention = ParamSpMM.AGNN_sddmm(d_output)
        if d_attention.dim() == 1:
            d_attention_exp = d_attention[None, :].expand(n_heads, -1)
        else:
            d_attention = d_attention.view(-1)
            d_attention = d_attention[ParamSpMM.mask]
            d_attention_exp = d_attention[None, :].expand(n_heads, -1)

        d_attention_w = torch.mm(d_attention_exp, colIdx.float()).transpose(0, 1)

        return (
            d_input,
            d_weights,
            d_attention_w,
            None,
            None,
        )


class AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_w = torch.nn.Parameter(torch.randn(1, n_heads))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, spmm_dict, colIdx):
        if self.weights.size(1) not in spmm_dict:
            raise ValueError("SpMMDecider not support dim: " + str(self.weights.size(1)))
        else:
            ParamSpMM = spmm_dict[self.weights.size(1)]
        return AGNNFunction.apply(X, self.weights, self.attention_w, ParamSpMM, colIdx)


class GINFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, ParamSpMM):
        # print("partSize: {}, dimWorker: {}, warpPerBlock: {}".format(inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock))
        X_agg = ParamSpMM.GIN_spmm(X)
        X_prime = torch.mm(X_agg, weight)
        ctx.save_for_backward(X_agg, weight)
        ctx.ParamSpMM = ParamSpMM
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        ParamSpMM = ctx.ParamSpMM
        X, weights = ctx.saved_tensors
        d_weights = torch.mm(X.t(), d_output)
        d_input_prime = torch.mm(d_output, weights.t())
        d_input = ParamSpMM.GIN_spmm(d_input_prime)

        return d_input, d_weights, None, None


class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, spmm_dict):
        if self.weights.size(1) not in spmm_dict:
            raise ValueError("SpMMDecider not support dim: " + str(self.weights.size(1)))
        else:
            ParamSpMM = spmm_dict[self.weights.size(1)]
        return GINFunction.apply(X, self.weights, ParamSpMM)

