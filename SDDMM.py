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


def compare(result, result_ref):
    if result is None or result_ref is None:
        raise ValueError("result or result_ref is None")
    error = torch.abs(result - result_ref)
    threshold = 1e-4 * torch.abs(result_ref)
    mask = error > threshold
    if mask.any() == False:
        print("# Verification PASSED")
    else:
        # print mismatched result
        j = 0
        for i in range(len(mask)):
            if mask[i] == True:
                print("mismatched result: ", i, result[i], result_ref[i])
                j += 1
            if j > 10:
                break
        raise ValueError("Verification FAILED")


def compare_2d(result, result_ref, mask):
    result = result.flatten().contiguous()
    mask = mask.to("cpu")
    result = result[mask]
    compare(result, result_ref)


class pcsr_sddmm(object):
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
                    self.mask = graph.mask.to(device)
            self.vec_size = vec_size
            self.if_split = if_split
            self.CF = CF
            self.blkWarpNum = blkWarpNum
        # result for verification
        self.result = None

    # CF is set to be dim//warp_size
    def get_sddmm_result(self, dim, blkWarpNum):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        A = torch.ones((self.mtx_nodes + 1) // 2 * 2, dim).to(device)
        B = torch.ones(self.mtx_nodes, dim).to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        CF = (dim + 31) // 32
        # print input info: shape dtype device
        # print("A: ", A.shape, A.dtype, A.device)
        # print("B: ", B.shape, B.dtype, B.device)
        # print("rowPtr: ", rowPtr.shape, rowPtr.dtype, rowPtr.device)
        # print("colIdx: ", colIdx.shape, colIdx.dtype, colIdx.device)
        # print("val: ", val.shape, val.dtype, val.device)
        # print("CF: ", CF)

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
        self.result = result.to("cpu")
        torch.cuda.empty_cache()

    def profiling(self, blkWarpNum, dim):
        """
        profiling for sddmm with first initialized CF and blkWarpNum
        return throughput
        """
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        A = torch.ones((self.mtx_nodes + 1) // 2 * 2, dim).to(device)
        B = torch.ones(self.mtx_nodes, dim).to(device)
        CF = (dim + 31) // 32
        if self.if_split == False:
            if self.vec_size == 1:
                throughput = SDDMM.csr_sddmm_profile(A, B, self.rowPtr, self.colIdx, self.val, CF, blkWarpNum)
            elif self.vec_size == 2:
                throughput = SDDMM.bcsr_sddmm_profile(
                    A,
                    B,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
            else:
                throughput = SDDMM.vec_sddmm_profile(
                    A,
                    B,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
        else:
            if self.vec_size == 1:
                throughput = SDDMM.csr_split_sddmm_profile(
                    A,
                    B,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    CF,
                    blkWarpNum,
                )
            elif self.vec_size == 2:
                throughput = SDDMM.bcsr_split_sddmm_profile(
                    A,
                    B,
                    self.rowPtr,
                    self.colIdx,
                    self.val,
                    self.target_row,
                    self.mtx_edges,
                    CF,
                    blkWarpNum,
                )
            else:
                throughput = SDDMM.vec_split_sddmm_profile(
                    A,
                    B,
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

    def cusparse_ref(self, dim=None, is_profiling=False):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        if self.if_split == True or self.vec_size != 1:
            raise ValueError("cusparse only support vec_size=1 and if_split=False")
        A = torch.ones((self.mtx_nodes + 1) // 2 * 2, dim).to(device)
        B = torch.ones(self.mtx_nodes, dim).to(device)
        rowPtr = self.rowPtr
        colIdx = self.colIdx
        val = self.val
        if is_profiling == True:
            throughput = SDDMM.cusparse_sddmm_profile(A, B, rowPtr, colIdx, val)
            print("cusparse-> dim: {} gflops: {:.0f}".format(dim, throughput))
            del A, B
            torch.cuda.empty_cache()
            return throughput
        else:
            result = SDDMM.cusparse_sddmm_compute(A, B, rowPtr, colIdx, val).to("cpu")
            del A, B
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


def profile_engine(filelist, path, dim_list, vec_size=2, if_auto_gran=True):
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

    for mtx_name in filelist:
        print("-------------{}-------------".format(mtx_name))
        graph = graph_input(mtx_name=mtx_name, path=path)
        graph.load(load_from_mtx=True, is_avg_granularity=(not if_auto_gran), blk_size=vec_size, if_mask=True)
        # coo_baseline = coo_spmm(graph)
        # <vec_size=1, if_split=False>
        Param10 = pcsr_sddmm(
            graph,
            vec_size=1,
            if_split=False,
        )
        # <vec_size=1, if_split=True>
        Param11 = pcsr_sddmm(
            graph,
            vec_size=1,
            if_split=True,
        )
        # <vec_size=2, if_split=False>
        Param20 = pcsr_sddmm(
            graph,
            vec_size=vec_size,
            if_split=False,
        )
        # <vec_size=2, if_split=True>
        Param21 = pcsr_sddmm(
            graph,
            vec_size=vec_size,
            if_split=True,
        )
        # baseline profile
        # for dim in dim_list:
        #     # varify cusparse_default result
        #     cusparse_result = Param10.cusparse_ref(dim)
        #     baseline_tp_dict[dim].append(Param10.cusparse_ref(dim, is_profiling=True))

        # profile ParamSpMM
        for dim in dim_list:
            cusparse_result = Param10.cusparse_ref(dim)
            for blkWarpNum in warpNum_list:
                Param10.get_sddmm_result(dim=dim, blkWarpNum=blkWarpNum)
                torch.cuda.synchronize()
                compare(Param10.result, cusparse_result)
                dim_tp_dict[dim].append(Param10.profiling(blkWarpNum, dim))
                Param11.get_sddmm_result(dim=dim, blkWarpNum=blkWarpNum)
                torch.cuda.synchronize()
                compare(Param11.result, cusparse_result)
                dim_tp_dict[dim].append(Param11.profiling(blkWarpNum, dim))
                Param20.get_sddmm_result(dim=dim, blkWarpNum=blkWarpNum)
                torch.cuda.synchronize()
                compare_2d(Param20.result, cusparse_result, Param20.mask)
                dim_tp_dict[dim].append(Param20.profiling(blkWarpNum, dim))
                Param21.get_sddmm_result(dim=dim, blkWarpNum=blkWarpNum)
                torch.cuda.synchronize()
                compare_2d(Param21.result, cusparse_result, Param21.mask)
                dim_tp_dict[dim].append(Param21.profiling(blkWarpNum, dim))
                torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # each dim has a corresponding dataframe
    TP_dataframe_dict = {}
    baseline_dataframe_dict = {}
    for dim in dim_list:
        np_size = (len(filelist), 4 * len(warpNum_list))
        tp_np = np.array(dim_tp_dict[dim]).reshape(np_size)
        # various data representations
        represent = ["vec1", "vec1_split", "vec" + str(vec_size), "vec" + str(vec_size) + "_split"]
        # various configurations
        configurations = []
        for blkWarpNum in warpNum_list:
            for re in represent:
                config = re + "_TP_" + str(blkWarpNum)
                configurations.append(config)
        data = {}
        for col, config in enumerate(configurations):
            data[config] = tp_np[..., col]
        TP = pd.DataFrame(data)
        TP_dataframe_dict[dim] = TP

        # baseline profiling dataframe
        # baseline_data = {}
        # baeline_tp = np.array(baseline_tp_dict[dim]).reshape((len(filelist), 2))
        # baseline_data["cusparse"] = baeline_tp[..., 0]
        # baseline_dataframe_dict[dim] = pd.DataFrame(baseline_data)

    return TP_dataframe_dict


# with DCP plugged in
def training_data_gen(dim_list, vec_size, info_df, filelist, path, csv_path):
    TP_dict = profile_engine(filelist, path, dim_list, vec_size, if_use_auto_gran)
    for dim in dim_list:
        TP = TP_dict[dim]
        TP.to_csv(csv_path + "dim" + str(dim) + ".csv", index=False)
        OP_SpMM = pd.DataFrame(
            TP.idxmax(axis=1, numeric_only=True).apply(lambda x: TP.columns.get_loc(x)),
            columns=["OP_SpMM"],
        )
        OP_SpMM = pd.concat([info_df, OP_SpMM], axis=1)
        OP_SpMM.to_csv(csv_path + "dim" + str(dim) + "_OP_SpMM.csv", index=False)

        # cusparse_df = cusparse_dict[dim]
        # cusparse_df.to_csv(csv_path + "dim" + str(dim) + "_baseline.csv", index=False)


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
        # default=[32],, 144, 160, 176, 192, 208, 224, 240, 256
        default=[32],
        help="dim_list for training data generation",
    )
    parser.add_argument("--mtx_name", type=str, default="", help="mtx file name")
    parser.add_argument("--vec_size", type=int, default=2, help="vec_size(besides 1) for SpMM")
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
    if_use_auto_gran = args.if_auto_gran == "True"
    mtx_name = args.mtx_name
    csv_path = "./ParamSpMM-log/OGBsddmm/"

    os.chdir(os.path.dirname(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filelist = []
    if mtx_name == "":
        for file in glob.glob(os.path.join(path, "*.mtx")):
            filename = os.path.basename(file)
            filelist.append(filename[:-4])
        filelist.sort()
    else:
        filelist.append(mtx_name)

    if not os.path.exists(csv_path + "mtx_info.csv"):
        info_df = get_mtx_info(filelist, path, if_auto_gran=if_use_auto_gran, blk_size=vec_size)
        info_df.to_csv(csv_path + "mtx_info.csv", index=False)
    else:
        info_df = pd.read_csv(csv_path + "mtx_info.csv")

    training_data_gen(dim_list, vec_size, info_df, filelist, path, csv_path)
