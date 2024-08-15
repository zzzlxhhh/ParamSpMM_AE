#!/usr/bin/env python3
import time
import argparse
import torch
import math
import numpy as np
import pandas as pd
import ParamSpMM as SpMM
import ParamSpCONV as SpCONV
import ParamSDDMM as SDDMM
from scipy.sparse import csr_array
import os
import glob
from verification import SpMM_verification
from graph_input import graph_input
from tqdm import *


def gap_calculator(dim, CF, warp_size=32):
    t_norm = min(dim, CF * warp_size)
    t_residue = dim % (warp_size * CF)
    # must be a multiple of CF * warp_size
    if t_residue == 0:
        return 0
    else:
        return t_norm - t_residue


# CF prompter
def dynamic_CF_prompter(dim, CF_set=None, warp_size=32):
    if CF_set == None:
        CF_set = [1, 2, 3, 4, 5, 6, 7, 8]
    CF_set = CF_set[: math.ceil(dim / 32)]
    valid_CF_set = []

    for CF in CF_set:
        gap = gap_calculator(dim, CF, warp_size)
        print("dim({})->CF:{} gap:{}".format(dim, CF, gap))
        if gap < 32:
            valid_CF_set.append(CF)
    CF_set = valid_CF_set
    return CF_set


class coo_spmm(object):
    def __init__(self, graph, device="cuda"):
        coo = csr_array(
            (graph.csr_data.val, graph.csr_data.colIdx, graph.csr_data.rowPtr),
            shape=(graph.mtx_nodes, graph.mtx_nodes),
        ).tocoo()

        self.row = torch.from_numpy(coo.row).to(device)
        self.col = torch.from_numpy(coo.col).to(device)
        self.val = torch.from_numpy(coo.data).to(device)
        self.mtx_nodes = graph.mtx_nodes

    def get_spmm_result(self, dim=None, x=None):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)

        result = SpMM.cusparse_compute_coo_alg4(X, self.row, self.col, self.val).to("cpu")
        del X
        torch.cuda.empty_cache()
        return result

    def profiling(self, dim=None, x=None):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        throughput = SpMM.cusparse_profile_coo_alg4(X, self.row, self.col, self.val)
        print("coo-> dim: {} gflops: {:.0f}".format(dim, throughput))
        return throughput


class csc_spmm(object):
    def __init__(self, graph, device="cuda"):
        csc = csr_array(
            (graph.csr_data.val, graph.csr_data.colIdx, graph.csr_data.rowPtr),
            shape=(graph.mtx_nodes, graph.mtx_nodes),
        ).tocsc()

        self.colPtr = torch.from_numpy(csc.indptr).to(device)
        self.rowIdx = torch.from_numpy(csc.indices).to(device)
        self.val = torch.from_numpy(csc.data).to(device)
        self.mtx_nodes = graph.mtx_nodes

    def get_spmm_result(self, dim=None, x=None):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)

        result = SpMM.cusparse_compute_csc(X, self.colPtr, self.rowIdx, self.val).to("cpu")
        del X
        torch.cuda.empty_cache()
        return result

    def profiling(self, dim=None, x=None):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        throughput = SpMM.cusparse_profile_csc(X, self.colPtr, self.rowIdx, self.val)
        print("csc-> dim: {} gflops: {:.0f}".format(dim, throughput))
        return throughput


class pcsr_spmm(object):
    def __init__(
        self,
        graph,
        vec_size,
        if_split,
        mtx_nodes=None,
        mtx_edges=None,
        rowPtr=None,
        colIdx=None,
        val=None,
        target_row=None,
        CF=1,
        blkWarpNum=4,
        device="cuda",
    ):
        """
        if graph != None: it will init with graph, CF and blkWarpNum
        else: init with mtx_nodes, mtx_edges, rowPtr, colIdx, val .etc
        !!! vec_size and if_split must be given
        mtx_nodes: number of nodes in the graph
        mtx_edges: number of edges in the graph
        """
        if graph == None:
            if mtx_edges == None or mtx_nodes == None or rowPtr == None or colIdx == None or val == None:
                raise ValueError("when graph is None, mtx_nodes, mtx_edges, rowPtr, colIdx, val should be given")
            self.mtx_nodes = mtx_nodes
            self.mtx_edges = mtx_edges
            self.rowPtr = rowPtr.to(device)
            self.colIdx = colIdx.to(device)
            self.val = val.to(device)
            if if_split == True and target_row == None:
                raise ValueError("target_row should be given when if_split is True")
            elif if_split == False and target_row != None:
                raise ValueError("target_row should not be given when if_split is False")
            else:
                self.target_row = target_row
            self.if_split = if_split
            if vec_size != 1 and val.shape[1] > 1:
                raise ValueError("when vec_size>1  val is 2D array")
            self.vec_size = vec_size
            self.CF = CF
            self.blkWarpNum = blkWarpNum
        else:
            self.mtx_nodes = graph.mtx_nodes
            self.mtx_edges = graph.mtx_edges
            self.mask = None
            if if_split == False:
                if vec_size == 1:
                    self.rowPtr = graph.csr_data.rowPtr.to(device)
                    self.colIdx = graph.csr_data.colIdx.to(device)
                    self.val = graph.csr_data.val.to(device)
                else:
                    self.rowPtr = graph.bcsr_data.rowPtr.to(device)
                    self.colIdx = graph.bcsr_data.colIdx.to(device)
                    self.val = graph.bcsr_data.val.to(device)
                    if graph.mask != None:
                        self.mask = graph.mask.to(device)
                self.target_row = None
            else:
                if vec_size == 1:
                    self.rowPtr = graph.csr_data.rowPtr_split.to(device)
                    self.colIdx = graph.csr_data.colIdx.to(device)
                    self.val = graph.csr_data.val.to(device)
                    self.target_row = graph.csr_data.target_row.to(device)
                else:
                    self.rowPtr = graph.bcsr_data.rowPtr_split.to(device)
                    self.colIdx = graph.bcsr_data.colIdx.to(device)
                    self.val = graph.bcsr_data.val.to(device)
                    self.target_row = graph.bcsr_data.target_row.to(device)
                    if graph.mask != None:
                        self.mask = graph.mask.to(device)
            self.vec_size = vec_size
            self.if_split = if_split
            self.CF = CF
            self.blkWarpNum = blkWarpNum
        # result for verification
        self.result = None

    def get_spmm_result(self, dim=None, x=None, CF=None, blkWarpNum=None):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        if CF == None:
            CF = self.CF
        if blkWarpNum == None:
            blkWarpNum = self.blkWarpNum
        if self.if_split == False:
            if self.vec_size == 1:
                result = SpMM.csr_spmm(X, rowPtr, colIdx, val, CF, blkWarpNum)
            elif self.vec_size == 2:
                result = SpMM.bcsr_spmm(X, rowPtr, colIdx, val, CF, blkWarpNum)[0 : self.mtx_nodes, :]
            else:
                result = SpMM.vec_spmm(X, rowPtr, colIdx, val, CF, blkWarpNum)[0 : self.mtx_nodes, :]
        else:
            if self.vec_size == 1:
                result = SpMM.csr_split_spmm(X, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)
            elif self.vec_size == 2:
                result = SpMM.bcsr_split_spmm(X, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)[
                    0 : self.mtx_nodes, :
                ]
            else:
                result = SpMM.vec_split_spmm(X, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)[
                    0 : self.mtx_nodes, :
                ]
        del X
        self.result = result.to("cpu")
        torch.cuda.empty_cache()

    def AGNN_spmm(self, spVal, X):
        if self.if_split == False:
            if self.vec_size == 1:
                result = SpMM.csr_spmm(X, self.rowPtr, self.colIdx, spVal, self.CF, self.blkWarpNum)
            elif self.vec_size == 2:
                result = SpMM.bcsr_spmm(X, self.rowPtr, self.colIdx, spVal, self.CF, self.blkWarpNum)
            else:
                result = SpMM.vec_spmm(X, self.rowPtr, self.colIdx, spVal, self.CF, self.blkWarpNum)
        else:
            if self.vec_size == 1:
                result = SpMM.csr_split_spmm(
                    X, self.rowPtr, self.colIdx, spVal, self.target_row, self.CF, self.blkWarpNum
                )
            elif self.vec_size == 2:
                result = SpMM.bcsr_split_spmm(
                    X, self.rowPtr, self.colIdx, spVal, self.target_row, self.CF, self.blkWarpNum
                )
            else:
                result = SpMM.vec_split_spmm(
                    X, self.rowPtr, self.colIdx, spVal, self.target_row, self.CF, self.blkWarpNum
                )
        return result

    def GIN_spmm(self, X):
        if self.if_split == False:
            if self.vec_size == 1:
                result = SpMM.csr_spmm(X, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
            elif self.vec_size == 2:
                result = SpMM.bcsr_spmm(X, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
            else:
                result = SpMM.vec_spmm(X, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
        else:
            if self.vec_size == 1:
                result = SpMM.csr_split_spmm(
                    X, self.rowPtr, self.colIdx, self.val, self.target_row, self.CF, self.blkWarpNum
                )
            elif self.vec_size == 2:
                result = SpMM.bcsr_split_spmm(
                    X, self.rowPtr, self.colIdx, self.val, self.target_row, self.CF, self.blkWarpNum
                )
            else:
                result = SpMM.vec_split_spmm(
                    X, self.rowPtr, self.colIdx, self.val, self.target_row, self.CF, self.blkWarpNum
                )
        return result

    def AGNN_sddmm(self, A):
        dim = A.size(1)
        B = A
        CF = (dim + 31) // 32
        val = self.val

        if self.if_split == False:
            if self.vec_size == 1:
                result = SDDMM.csr_sddmm(A, B, self.rowPtr, self.colIdx, val, CF, self.blkWarpNum)
            elif self.vec_size == 2:
                if self.mask == None:
                    raise ValueError("mask should be given when vec_size=2")
                result = SDDMM.bcsr_sddmm(A, B, self.rowPtr, self.colIdx, val, CF, self.blkWarpNum)
            else:
                result = SDDMM.vec_sddmm(A, B, self.rowPtr, self.colIdx, val, CF, self.blkWarpNum)
        else:
            if self.vec_size == 1:
                result = SDDMM.csr_split_sddmm(
                    A, B, self.rowPtr, self.colIdx, val, self.target_row, CF, self.blkWarpNum
                )
            elif self.vec_size == 2:
                if self.mask == None:
                    raise ValueError("mask should be given when vec_size=2")
                result = SDDMM.bcsr_split_sddmm(
                    A, B, self.rowPtr, self.colIdx, val, self.target_row, CF, self.blkWarpNum
                )
            else:
                result = SDDMM.vec_split_sddmm(
                    A, B, self.rowPtr, self.colIdx, val, self.target_row, CF, self.blkWarpNum
                )
        return result

    def get_sddmm_result(self, dim, blkWarpNum):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        A = torch.ones((self.mtx_nodes + 1) // 2 * 2, dim).to(device)
        B = torch.ones(self.mtx_nodes, dim).to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        CF = (dim + 31) // 32

        if self.if_split == False:
            if self.vec_size == 1:
                result = SDDMM.csr_sddmm(A, B, rowPtr, colIdx, val, CF, blkWarpNum)
            elif self.vec_size == 2:
                result = SDDMM.bcsr_sddmm(A, B, rowPtr, colIdx, val, CF, blkWarpNum)
            else:
                result = SDDMM.vec_sddmm(A, B, rowPtr, colIdx, val, CF, blkWarpNum)
        else:
            if self.vec_size == 1:
                result = SDDMM.csr_split_sddmm(A, B, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)
            elif self.vec_size == 2:
                result = SDDMM.bcsr_split_sddmm(A, B, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)
            else:
                result = SDDMM.vec_split_sddmm(A, B, rowPtr, colIdx, val, self.target_row, CF, blkWarpNum)
        del A, B

    def forward(self, H, w):
        if H == None or w == None:
            raise ValueError("H and w should be given")
        if self.if_split == False:
            if self.vec_size == 1:
                result = SpCONV.csr_spmm_forward(H, w, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
            else:
                result = SpCONV.bcsr_spmm_forward(H, w, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
        else:
            if self.vec_size == 1:
                result = SpCONV.csr_split_spmm_forward(
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.CF,
                    self.blkWarpNum,
                )
            else:
                result = SpCONV.bcsr_split_spmm_forward(
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.CF,
                    self.blkWarpNum,
                )
        return result

    def backward(self, d_output, H, w):
        if d_output == None or H == None:
            raise ValueError("d_output and H should be given")
        if self.if_split == False:
            if self.vec_size == 1:
                d_input, d_weight = SpCONV.csr_spmm_backward(
                    d_output,
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.CF,
                    self.blkWarpNum,
                )
            else:
                d_input, d_weight = SpCONV.bcsr_spmm_backward(
                    d_output,
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.CF,
                    self.blkWarpNum,
                )
        else:
            if self.vec_size == 1:
                d_input, d_weight = SpCONV.csr_split_spmm_backward(
                    d_output,
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.CF,
                    self.blkWarpNum,
                )
            else:
                d_input, d_weight = SpCONV.bcsr_split_spmm_backward(
                    d_output,
                    H,
                    w,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.CF,
                    self.blkWarpNum,
                )
        return d_input, d_weight

    def cf_warpnum_profiling(self, CF, blkWarpNum, dim=None, x=None):
        """
        define CF and blkWarpNum for profiling
        (not the CF and blkWarpNum in __init__)
        """
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
            dim = x.shape[1]

        if self.if_split == False:
            if self.vec_size == 1:
                throughput = SpMM.csr_spmm_profile(X, self.rowPtr, self.colIdx, self.val, CF, blkWarpNum)
            elif self.vec_size == 2:
                throughput = SpMM.bcsr_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
            else:
                throughput = SpMM.vec_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
        else:
            if self.vec_size == 1:
                throughput = SpMM.csr_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    CF,
                    blkWarpNum,
                )
            elif self.vec_size == 2:
                throughput = SpMM.bcsr_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
            else:
                throughput = SpMM.vec_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )

        print("dim: {} gflops: {:.0f}".format(dim, throughput))
        return throughput

    def profiling(self, dim=None, x=None):
        """
        profiling for SpMM with first initialized CF and blkWarpNum
        return throughput
        """
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
            dim = x.shape[1]

        if self.if_split == False:
            if self.vec_size == 1:
                throughput = SpMM.csr_spmm_profile(X, self.rowPtr, self.colIdx, self.val, self.CF, self.blkWarpNum)
            elif self.vec_size == 2:
                throughput = SpMM.bcsr_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    self.CF,
                    self.blkWarpNum,
                )
            else:
                throughput = SpMM.vec_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    self.CF,
                    self.blkWarpNum,
                )
        else:
            if self.vec_size == 1:
                throughput = SpMM.csr_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.CF,
                    self.blkWarpNum,
                )
            elif self.vec_size == 2:
                throughput = SpMM.bcsr_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.mtx_edges,
                    self.CF,
                    self.blkWarpNum,
                )
            else:
                throughput = SpMM.vec_split_spmm_profile(
                    X,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.mtx_edges,
                    self.CF,
                    self.blkWarpNum,
                )

        print("dim: {} gflops: {:.0f}".format(dim, throughput))
        return throughput

    def cusparse_ref(self, dim=None, x=None, is_profiling=False):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if self.if_split == True or self.vec_size != 1:
            raise ValueError("cusparse only support vec_size=1 and if_split=False")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        if is_profiling == True:
            throughput = SpMM.cusparse_profile(X, rowPtr, colIdx, val)
            print("cusparse-> dim: {} gflops: {:.0f}".format(dim, throughput))
            del X
            torch.cuda.empty_cache()
            return throughput
        else:
            result = SpMM.cusparse_compute(X, rowPtr, colIdx, val).to("cpu")
            del X
            torch.cuda.empty_cache()
            return result

    def cusparse_csr_alg2_ref(self, dim=None, x=None, is_profiling=False):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if self.if_split == True or self.vec_size != 1:
            raise ValueError("cusparse only support vec_size=1 and if_split=False")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        if is_profiling == True:
            throughput = SpMM.cusparse_profile_csr_alg2(X, self.rowPtr, self.colIdx, self.val)
            print("cusparse_csr_alg2-> dim: {} gflops: {:.0f}".format(dim, throughput))
            del X
            torch.cuda.empty_cache()
            return throughput
        else:
            result = SpMM.cusparse_compute_csr_alg2(X, self.rowPtr, self.colIdx, self.val).to("cpu")
            del X
            torch.cuda.empty_cache()
            return result

    def gespmm_ref(self, dim=None, x=None, is_profiling=False):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if self.if_split == True or self.vec_size != 1:
            raise ValueError("cusparse only support vec_size=1 and if_split=False")
        if x == None and dim == None:
            raise ValueError("at least one of x and dim should be given")
        elif dim != None:
            X = torch.ones(self.mtx_nodes, dim).to(device)
        else:
            X = x.to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        if is_profiling == True:
            throughput = SpMM.gespmm_profile(X, rowPtr, colIdx, val)
            print("gespmm-> dim: {} gflops: {:.0f}".format(dim, throughput))
            del X
            torch.cuda.empty_cache()
            return throughput
        else:
            result = SpMM.gespmm_compute(X, rowPtr, colIdx, val).to("cpu")
            del X
            torch.cuda.empty_cache()
            return result


# get mtx info in pandas dataframe
def get_mtx_info(filelist, path, dim=32, if_auto_gran=True, blk_size=2):
    info_pd = pd.DataFrame()
    for mtx_name in filelist:
        graph = graph_input(mtx_name=mtx_name, path=path)
        graph.load(load_from_mtx=True, is_avg_granularity=(not if_auto_gran), blk_size=blk_size)
        # if graph.mtx_nodes * dim * 4 > 6 * (2**30):
        #     continue  # skip large mtx
        info_pd = pd.concat([info_pd, graph.mtxinfo_to_pandas()], axis=0, ignore_index=True)
    return info_pd


def profile_engine(filelist, path, dim_list, vec_size=2, if_DCP=True, if_auto_gran=True):
    """
    dim is a list
    avoid reading graph multiple times
    """

    warpNum_list = [4, 8]
    # init tp_list for each dim
    dim_tp_dict = {}
    baseline_tp_dict = {}
    for dim in dim_list:
        dim_tp_dict[dim] = []
        baseline_tp_dict[dim] = []
    # generate CF_list for each dim
    dim_CF_dict = {}
    origin_CF_list = [1, 2, 3, 4, 5, 6, 7, 8]
    for dim in dim_list:
        if if_DCP:
            dim_CF_dict[dim] = dynamic_CF_prompter(dim, origin_CF_list)
        else:
            dim_CF_dict[dim] = origin_CF_list[: math.ceil(dim / 32)]

    for mtx_name in filelist:
        print("-------------{}-------------".format(mtx_name))
        graph = graph_input(mtx_name=mtx_name, path=path)
        graph.load(load_from_mtx=True, is_avg_granularity=(not if_auto_gran), blk_size=vec_size)
        spmm_veri = SpMM_verification(
            dim,
            graph.csr_data.rowPtr,
            graph.csr_data.colIdx,
            graph.csr_data.val,
            baseline=True,
        )
        # coo_baseline = coo_spmm(graph)
        # <vec_size=1, if_split=False>
        Param10 = pcsr_spmm(
            graph,
            vec_size=1,
            if_split=False,
        )
        # <vec_size=1, if_split=True>
        Param11 = pcsr_spmm(
            graph,
            vec_size=1,
            if_split=True,
        )
        # <vec_size=2, if_split=False>
        Param20 = pcsr_spmm(
            graph,
            vec_size=vec_size,
            if_split=False,
        )
        # <vec_size=2, if_split=True>
        Param21 = pcsr_spmm(
            graph,
            vec_size=vec_size,
            if_split=True,
        )
        # baseline profile
        for dim in dim_list:
            spmm_veri.reference(dim)
            # varify cusparse_default result
            cusparse_result = Param10.cusparse_ref(dim)
            spmm_veri.compare(cusparse_result, mtx_name)

            # varify cusparse_csr_alg2 result
            # cusparse_csr_alg2_result = Param10.cusparse_csr_alg2_ref(dim)
            # spmm_veri.compare(cusparse_csr_alg2_result, mtx_name)

            # varify cusparse_coo_alg4 result
            # cusparse_coo_alg4_result = coo_baseline.get_spmm_result(dim)
            # spmm_veri.compare(cusparse_coo_alg4_result, mtx_name)

            # varify gespmm result
            gespmm_result = Param10.gespmm_ref(dim)
            spmm_veri.compare(gespmm_result, mtx_name)
            baseline_tp_dict[dim].append(Param10.cusparse_ref(dim, is_profiling=True))
            # baseline_tp_dict[dim].append(Param10.cusparse_csr_alg2_ref(dim, is_profiling=True))
            # baseline_tp_dict[dim].append(coo_baseline.profiling(dim))
            baseline_tp_dict[dim].append(Param10.gespmm_ref(dim, is_profiling=True))

        # profile ParamSpMM
        for dim in dim_list:
            CF_list = dim_CF_dict[dim]
            # spmm_veri.reference(dim)
            for blkWarpNum in warpNum_list:
                for cf in CF_list:
                    Param10.get_spmm_result(dim=dim, CF=cf, blkWarpNum=blkWarpNum)
                    spmm_veri.compare(Param10.result, mtx_name)
                    dim_tp_dict[dim].append(Param10.cf_warpnum_profiling(cf, blkWarpNum, dim))
                    Param11.get_spmm_result(dim=dim, CF=cf, blkWarpNum=blkWarpNum)
                    spmm_veri.compare(Param11.result, mtx_name)
                    dim_tp_dict[dim].append(Param11.cf_warpnum_profiling(cf, blkWarpNum, dim))
                    Param20.get_spmm_result(dim=dim, CF=cf, blkWarpNum=blkWarpNum)
                    spmm_veri.compare(Param20.result, mtx_name)
                    dim_tp_dict[dim].append(Param20.cf_warpnum_profiling(cf, blkWarpNum, dim))
                    Param21.get_spmm_result(dim=dim, CF=cf, blkWarpNum=blkWarpNum)
                    spmm_veri.compare(Param21.result, mtx_name)
                    dim_tp_dict[dim].append(Param21.cf_warpnum_profiling(cf, blkWarpNum, dim))
        torch.cuda.empty_cache()

    # each dim has a corresponding dataframe
    TP_dataframe_dict = {}
    baseline_dataframe_dict = {}
    for dim in dim_list:
        CF_list = dim_CF_dict[dim]
        np_size = (len(filelist), 4 * len(warpNum_list) * len(CF_list))
        tp_np = np.array(dim_tp_dict[dim]).reshape(np_size)
        # various data representations
        represent = ["vec1", "vec1_split", "vec" + str(vec_size), "vec" + str(vec_size) + "_split"]
        # various configurations
        configurations = []
        for blkWarpNum in warpNum_list:
            for cf in CF_list:
                for re in represent:
                    config = re + "_TP_" + str(cf) + str(blkWarpNum)
                    configurations.append(config)
        data = {}
        for col, config in enumerate(configurations):
            data[config] = tp_np[..., col]
        TP = pd.DataFrame(data)
        TP_dataframe_dict[dim] = TP

        # baseline profiling dataframe
        baseline_data = {}
        baeline_tp = np.array(baseline_tp_dict[dim]).reshape((len(filelist), 2))
        baseline_data["cusparse"] = baeline_tp[..., 0]
        # baseline_data["cusparse_csr_alg2"] = baeline_tp[..., 1]
        # baseline_data["cusparse_coo_alg4"] = baeline_tp[..., 2]
        # baseline_data["gespmm"] = baeline_tp[..., 3]
        baseline_data["gespmm"] = baeline_tp[..., 1]

        baseline_dataframe_dict[dim] = pd.DataFrame(baseline_data)

    return TP_dataframe_dict, baseline_dataframe_dict


# with DCP plugged in
def training_data_gen(dim_list, vec_size, info_df, filelist, path, csv_path):
    TP_dict, cusparse_dict = profile_engine(filelist, path, dim_list, vec_size, if_use_DCP, if_use_auto_gran)
    for dim in dim_list:
        TP = TP_dict[dim]
        TP.to_csv(csv_path + "dim" + str(dim) + ".csv", index=False)
        OP_SpMM = pd.DataFrame(
            TP.idxmax(axis=1, numeric_only=True).apply(lambda x: TP.columns.get_loc(x)),
            columns=["OP_SpMM"],
        )
        OP_SpMM = pd.concat([info_df, OP_SpMM], axis=1)
        OP_SpMM.to_csv(csv_path + "dim" + str(dim) + "_OP_SpMM.csv", index=False)

        cusparse_df = cusparse_dict[dim]
        cusparse_df.to_csv(csv_path + "dim" + str(dim) + "_baseline.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./toy_dataset/",
        help="path to the mtx files",
    )
    parser.add_argument(
        "--dim_list",
        nargs="*",
        type=int,
        # default=[32],,16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256
        default=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256],
        help="dim_list for training data generation",
    )
    parser.add_argument("--vec_size", type=int, default=2, help="vec_size(besides 1) for SpMM")

    parser.add_argument(
        "--if_DCP",
        type=str,
        choices=["True", "False"],
        default="True",
        help="If to use DCP",
    )
    parser.add_argument(
        "--if_reorder",
        type=str,
        choices=["True", "False"],
        default="True",
        help="If to use graph reordering",
    )
    parser.add_argument(
        "--if_auto_gran",
        type=str,
        choices=["True", "False"],
        default="True",
        help="If to use auto-Granularity, else use avg_row_nnz as split-granularity",
    )
    parser.add_argument(
        "--approx_eval",
        type=str,
        choices=["True", "False"],
        default="False",
        help="case study for approximate model",
    )

    args = parser.parse_args()
    print(args)
    path = args.path
    dim_list = args.dim_list
    vec_size = args.vec_size
    print("dim_list: ", dim_list)
    if_use_DCP = args.if_DCP == "True"
    if_use_auto_gran = args.if_auto_gran == "True"
    if_use_reorder = args.if_reorder == "True"
    approx_eval = args.approx_eval == "True"

    if if_use_DCP == True and if_use_auto_gran == True:
        csv_path = "./ParamSpMM-log/"
    elif if_use_DCP == True and if_use_auto_gran == False:
        csv_path = "./ParamSpMM-log/ablation_study/auto-Gran/"
    elif if_use_DCP == False and if_use_auto_gran == True:
        csv_path = "./ParamSpMM-log/ablation_study/woDCP/"
    else:
        raise ValueError("input if_DCP and if_auto_gran options are not supported")

    if if_use_reorder:
        assert "rabbit" in path
    else:
        assert path.find("rabbit") == -1
        csv_path = "./ParamSpMM-log/ablation_study/wo_reorder/"

    if vec_size != 2:
        csv_path = "./ParamSpMM-log/ablation_study/vec_size" + str(vec_size) + "/"

    if approx_eval == True:
        csv_path = "./ParamSpMM-log/ablation_study/approx_eval/"
        dim_list = [8, 24, 40, 56, 72, 88, 104, 120]
        # dim_list = [136, 152, 168, 184, 200, 216, 232]
    os.chdir(os.path.dirname(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filelist = []
    for file in glob.glob(os.path.join(path, "*.mtx")):
        filename = os.path.basename(file)
        filelist.append(filename[:-4])
    filelist.sort()
    # filelist = filelist[:3]

    if not os.path.exists(csv_path + "mtx_info.csv"):
        info_df = get_mtx_info(filelist, path, if_auto_gran=if_use_auto_gran, blk_size=vec_size)
        info_df.to_csv(csv_path + "mtx_info.csv", index=False)
    else:
        info_df = pd.read_csv(csv_path + "mtx_info.csv")

    training_data_gen(dim_list, vec_size, info_df, filelist, path, csv_path)
