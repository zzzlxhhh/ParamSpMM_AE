#!/usr/bin/env python3
import time
import torch
import os
import glob
import math
import pandas as pd
import ParamSpMM as SpMM
import ParamSDDMM as SDDMM
from scipy.io import mmread
from scipy.sparse import csr_array
from scipy.sparse import diags
import numpy as np
from tqdm import tqdm


class csr_data(object):
    def __init__(self, rowPtr, colIdx, val, num_nodes, num_edges, csr_flag=True):
        self.csr_flag = csr_flag
        # data in csr format in torch tensor
        self.rowPtr = torch.from_numpy(rowPtr).int()
        self.colIdx = torch.from_numpy(colIdx).int()
        self.val = torch.from_numpy(val).float()
        self.row_nnz = None
        # mtx size info
        self.num_nodes = num_nodes  # size of rowPtr in csr/bcsr
        self.num_edges = num_edges  # number of val(blk or scalar) in bcsr/csr
        # self.density = None
        self.max_row_nnz = None
        self.real_nrow = None
        self.nonempty_rate = None
        # mtx statistics info
        self.avg_degree = None
        self.COV = None
        # mtx statistics info without empty rows
        self.real_avg_degree = None
        self.real_COV = None
        # locality info
        self.max_bw = None
        self.avg_bw = None
        # split info
        self.split_granularity = None
        self.split_ratio = None
        # split data
        self.rowPtr_split = None
        self.target_row = None

    def split_ratio_comp(self, split_granularity):
        row_split_nnz = torch.ceil(self.row_nnz / split_granularity)
        split_ratio = int(row_split_nnz.sum()) / self.real_nrow
        return split_ratio, row_split_nnz

    def init_info(self, manual_granularity=None, is_avg_granularity=False):
        """
        manual_granularity: use input manual_granularity as split_granularity
        is_avg_granularity: use real_avg_degree as split_granularity
        if manual_granularity is None and is_avg_granularity is False:
            use auto-granularity
        """
        if self.csr_flag == True:
            # size info
            row_nnz = self.rowPtr[1:] - self.rowPtr[:-1]
            self.row_nnz = row_nnz
            self.max_row_nnz = torch.max(row_nnz)
            self.real_nrow = torch.sum(row_nnz != 0)
            self.nonempty_rate = self.real_nrow / self.num_nodes
            non_empty_idx = [row_nnz != 0]
            # mtx statistics info
            self.avg_degree = self.num_edges / self.num_nodes
            self.COV = row_nnz.float().std() / self.avg_degree
            # mtx statistics info without empty rows
            if self.nonempty_rate < 0.98:
                self.real_avg_degree = (self.num_edges / self.real_nrow).item()
                self.real_COV = torch.std(row_nnz[non_empty_idx].float()) / self.real_avg_degree
            else:
                self.real_avg_degree = self.avg_degree
                self.real_COV = self.COV
            # locality info
            tail = self.rowPtr[1:] - 1
            tail = tail[non_empty_idx]
            head = self.rowPtr[:-1]
            head = head[non_empty_idx]
            bw = self.colIdx[tail.long()] - self.colIdx[head.long()]
            # indices = row_nnz.where(row_nnz != 0)
            self.max_bw = torch.max(bw)
            bw_sum = torch.sum(bw)
            self.avg_bw = bw_sum / self.real_nrow
            # split
            if is_avg_granularity == True:
                self.split_granularity = int(self.real_avg_degree)
                self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)
            elif manual_granularity != None:
                self.split_granularity = manual_granularity
                self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)
            # auto granularity
            else:
                # gran = int(self.real_avg_degree)
                # if gran < 16:
                #     self.split_granularity = 16
                # else:
                self.split_granularity = math.ceil(self.real_avg_degree / 32) * 32
                self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)
        else:
            # bcsr only need to get split info,  bw or max_bw is not needed
            self.get_split_info(manual_granularity, is_avg_granularity)

    def get_split_info(self, manual_granularity=None, is_avg_granularity=False):
        """
        get split_granularity and split_ratio
        is_avg_granularity: use real_avg_degree as split_granularity
        manual_granularity: use manual_granularity as split_granularity
        """
        # if self.split_granularity == None:
        self.row_nnz = self.rowPtr[1:] - self.rowPtr[:-1]
        self.real_nrow = torch.sum(self.row_nnz != 0)
        self.real_avg_degree = (self.num_edges / self.real_nrow).item()
        if is_avg_granularity == True:
            self.split_granularity = int(self.real_avg_degree)
            self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)
        elif manual_granularity != None:
            self.split_granularity = manual_granularity
            self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)
        else:
            # gran = int(self.real_avg_degree)
            # if gran < 16:
            #     self.split_granularity = 16
            # else:
            self.split_granularity = math.ceil(self.real_avg_degree / 32) * 32
            self.split_ratio, self.row_split_nnz = self.split_ratio_comp(self.split_granularity)

    def get_split(self):
        if self.split_granularity == None:
            raise ValueError("split_granularity is None, must be init first")
        start = time.perf_counter()
        row_split_offset = torch.cat((torch.tensor([0]), self.row_split_nnz.cumsum(dim=0))).type(torch.int32)
        self.rowPtr_split, self.target_row = SpMM.split_csr_par(self.rowPtr, row_split_offset, self.split_granularity)
        # self.rowPtr_split, self.target_row = SpMM.split_csr(self.rowPtr, self.split_granularity)
        dur = time.perf_counter() - start
        print(
            "get_split -> split time: (ms): {:.3f} split_gran: {} \n split_ratio: {} avg_nnz: {} split_rows: {}".format(
                dur * 1e3,
                self.split_granularity,
                self.split_ratio,
                self.real_avg_degree,
                self.target_row.shape[0],
            )
        )
        return

    def print_info(self):
        if self.csr_flag == True:
            print("#################### CSR format graph info ####################")
            print("= num_nodes: {}".format(self.num_nodes))
            print("= num_edges: {}".format(self.num_edges))
            # print("= density: {:.5f}".format(self.density))
            print("= max_row_nnz: {}".format(self.max_row_nnz))
            print("= real_nrow: {}".format(self.real_nrow))
            print("= nonempty_rate: {:.3f}".format(self.nonempty_rate))
            print("#################### mtx statistics info ####################")
            print("= avg_degree: {:.3f}".format(self.avg_degree))
            print("= COV: {:.3f}".format(self.COV))
            print("########## mtx statistics info without empty rows ##########")
            print("= real_avg_degree: {:.3f}".format(self.real_avg_degree))
            print("= real_COV: {:.3f}".format(self.real_COV))
            print("##################### locality info #####################")
            print("= max_bw: {}".format(self.max_bw))
            print("= avg_bw: {:.3f}".format(self.avg_bw))
            # print("# real_avg_bw: {:.3f}".format(self.real_avg_bw))
            print("###################### split info ######################")
            print("= split_granularity: {}".format(self.split_granularity))
            print("= split_ratio: {:.5f}".format(self.split_ratio))
        else:
            print("#################### BCSR format graph info ####################")
            print("= real_nrow: {}".format(self.real_nrow))
            print("= real_avg_degree: {:.3f}".format(self.real_avg_degree))
            print("= split_granularity: {}".format(self.split_granularity))
            print("= split_ratio: {:.5f}".format(self.split_ratio))


class graph_input(object):
    def __init__(self, mtx_name=None, path=None):
        self.loaded_flag = False
        self.reordered_flag = False
        self.csr_split_flag = False
        self.bcsr_split_flag = False
        self.bcsr_flag = False

        self.path = path
        self.mtx_name = mtx_name
        """
        graph info used for classification
        """
        # mtx size info
        self.mtx_nodes = None
        self.mtx_edges = None
        self.density = None

        self.csr_data = None
        self.bcsr_data = None
        # self.reuse_rate = None
        self.pack_rate = None

    def load(
        self,
        load_from_mtx=True,
        manual_granularity=None,
        is_avg_granularity=False,
        is_normalize=False,
        GIN_epsilon=None,
        blk_size=2,
        if_mask=False,
    ):
        """
        load the graph from the disk --> CPU memory.
        """
        if self.path == None:
            raise ValueError("Graph path must be assigned first")

        if load_from_mtx:
            """
            edge in the txt format:
            s0 d0
            s1 d1
            s2 d2
            """
            mtx = mmread(self.path + self.mtx_name + ".mtx")
            mtx.data = np.ones_like(mtx.data)
            if is_normalize:
                mtx = self.normalize_adj(mtx)
            elif GIN_epsilon != None:
                mtx = self.normalize_adj_GIN(mtx, GIN_epsilon)
            csr = mtx.tocsr()
            start = time.perf_counter()
            # csr data
            self.mtx_nodes = csr.shape[0]
            self.mtx_edges = csr.nnz
            self.density = self.mtx_edges / (self.mtx_nodes * self.mtx_nodes)
            self.csr_data = csr_data(csr.indptr, csr.indices, csr.data, self.mtx_nodes, self.mtx_edges, True)
            start_csr = time.perf_counter()

            self.csr_data.init_info(is_avg_granularity=is_avg_granularity, manual_granularity=manual_granularity)
            print("csr init time: {:.3f}".format((time.perf_counter() - start_csr) * 1e3))
            # bcsr data
            start_bcsr = time.perf_counter()
            if self.mtx_nodes % blk_size == 0:
                bcsr = csr.tobsr(blocksize=(blk_size, 1))
            else:
                padded_rowPtr = csr.indptr
                residue = blk_size - (self.mtx_nodes % blk_size)
                for i in range(residue):
                    padded_rowPtr = np.concatenate((padded_rowPtr, [self.mtx_edges]))
                padded_csr = csr_array(
                    (csr.data, csr.indices, padded_rowPtr),
                    shape=(self.mtx_nodes + residue, self.mtx_nodes + residue),
                )
                bcsr = padded_csr.tobsr(blocksize=(blk_size, 1))

            nnzblk = bcsr.nnz / blk_size
            self.bcsr_data = csr_data(
                bcsr.indptr,
                bcsr.indices,
                bcsr.data.squeeze(),
                int(bcsr.shape[0] / blk_size),
                nnzblk,
                False,
            )
            # if is_avg_granularity:
            #     self.bcsr_data.init_info(is_avg_granularity=True)
            # elif manual_granularity != None:
            #     self.bcsr_data.init_info(manual_granularity)
            # else:
            self.bcsr_data.init_info(is_avg_granularity=is_avg_granularity, manual_granularity=manual_granularity)
            # self.reuse_rate = (self.mtx_edges - nnzblk) / nnzblk
            self.pack_rate = self.mtx_edges / bcsr.nnz
            print("bcsr init time: {:.3f}".format((time.perf_counter() - start_bcsr) * 1e3))
            self.mask = None
            if if_mask:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("mtx_nodes:{}, mtx_edges:{}".format(self.mtx_nodes, self.mtx_edges))
                mask = SDDMM.genmask(
                    self.bcsr_data.rowPtr.to(device),
                    self.csr_data.rowPtr.to(device),
                    self.bcsr_data.val.to(device),
                    self.mtx_edges,
                )
                self.mask = mask.to(torch.int64)

        self.loaded_flag = True
        self.reordered_flag = True
        self.csr_split_flag = False
        self.bcsr_split_flag = False
        self.bcsr_flag = True
        self.get_split_csr()
        self.get_split_bcsr()
        dur = time.perf_counter() - start
        print("data format all preprocess (ms): {:.3f}".format(dur * 1e3))
        return

    def get_split_csr(self):
        self.csr_split_flag = True
        self.csr_data.get_split()
        # use tensor to compute the split info(split_ratio & split_rows)
        # self.csr_data.get_split()
        return

    def get_split_bcsr(self):
        self.bcsr_split_flag = True
        self.bcsr_data.get_split()

    def decision_tree_32():
        return

    def decision_tree_64():
        return

    def decision_tree_128():
        return

    def print_mtx_info(self):
        """
        print the graph info
        """
        print("-----------------{}-----------------".format(mtx_name))
        print("= graph_nodes: {}".format(self.mtx_nodes))
        print("= graph_edges: {}".format(self.mtx_edges))
        print("= graph_density: {}".format(self.density))
        # print("= reuse_rate: {:.5f}".format(self.reuse_rate))
        print("= pack_rate: {:.5f}".format(self.pack_rate))
        self.csr_data.print_info()
        self.bcsr_data.print_info()

    def mtxinfo_to_pandas(self):
        data = {
            "mtx_name": self.mtx_name,
            "graph_nodes": self.mtx_nodes,
            "graph_edges": self.mtx_edges,
            "graph_density": self.density,
            # csr info
            "csr_max_row_nnz": self.csr_data.max_row_nnz.item(),  # max_degree
            "csr_real_nrow": self.csr_data.real_nrow.item(),
            "csr_nonempty_rate": self.csr_data.nonempty_rate.item(),
            "csr_avg_degree": self.csr_data.avg_degree,
            "csr_COV": self.csr_data.COV.item(),
            "csr_real_avg_degree": self.csr_data.real_avg_degree,
            "csr_real_COV": self.csr_data.real_COV.item(),
            "csr_max_bw": self.csr_data.max_bw.item(),
            "csr_avg_bw": self.csr_data.avg_bw.item(),
            "csr_split_ratio": self.csr_data.split_ratio.item(),
            # bcsr info
            # "bcsr_max_row_nnz": self.bcsr_data.max_row_nnz.item(),  # max_degree
            # "bcsr_real_nrow": self.bcsr_data.real_nrow.item(),
            # "bcsr_nonempty_rate": self.bcsr_data.nonempty_rate.item(),
            # "bcsr_avg_degree": self.bcsr_data.avg_degree,
            # "bcsr_COV": self.bcsr_data.COV.item(),
            # "bcsr_real_avg_degree": self.bcsr_data.real_avg_degree,
            # "bcsr_real_COV": self.bcsr_data.real_COV.item(),
            # "bcsr_max_bw": self.bcsr_data.max_bw.item(),
            # "bcsr_avg_bw": self.bcsr_data.avg_bw.item(),
            "bcsr_split_ratio": self.bcsr_data.split_ratio.item(),
            "pack_rate": self.pack_rate,
        }
        info_pd = pd.DataFrame(data, index=[0])
        # print(info_pd)
        return info_pd

    def normalize_adj(self, mtx):
        """
        first make all nz to be 1
        Symmetrically normalize adjacency matrix.
        """
        mtx.data = np.ones_like(mtx.data)
        mtx.setdiag(1)
        degree = np.array(mtx.sum(1))
        # must get flatterned for diags(input)
        d_inv_sqrt = np.power(degree, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = diags(d_inv_sqrt)
        tmp = mtx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return tmp

    def normalize_adj_GIN(self, mtx, epsilon):
        """
        first make all nz to be 1
        Symmetrically normalize adjacency matrix.
        """
        mtx.data = np.ones_like(mtx.data)
        mtx.setdiag(1 + epsilon)
        tmp = (mtx + mtx.T) / 2
        # tmp.data = np.ones_like(tmp.data)
        return tmp

    def get_info(self):
        data = [
            self.mtx_nodes,
            self.mtx_edges,
            self.density,
            self.csr_data.max_row_nnz.item(),
            self.csr_data.real_nrow.item(),
            self.csr_data.nonempty_rate.item(),
            self.csr_data.avg_degree,
            self.csr_data.COV.item(),
            self.csr_data.real_avg_degree,
            self.csr_data.real_COV.item(),
            self.csr_data.max_bw.item(),
            self.csr_data.avg_bw.item(),
            self.csr_data.split_ratio.item(),
            self.bcsr_data.split_ratio.item(),
            self.pack_rate,
        ]
        return np.array(data).reshape(1, -1)

    def init_feat_embedding(self, num_features, vec_size=None):
        if num_features == None:
            raise ValueError("num_features must be assigned first")
        self.num_features = num_features
        if vec_size == None:
            self.x = torch.randn(self.mtx_nodes, num_features).cuda()
        else:
            rows = (self.mtx_nodes + vec_size - 1) // vec_size * vec_size
            self.x = torch.randn(rows, num_features).cuda()

    def init_labels(self, num_classes, vec_size=None):
        if num_classes == None:
            raise ValueError("num_classes must be assigned first")
        self.num_classes = num_classes
        if vec_size == None:
            self.y = torch.ones(self.mtx_nodes).long().cuda()
        else:
            rows = (self.mtx_nodes + vec_size - 1) // vec_size * vec_size
            self.y = torch.ones(rows).long().cuda()


if __name__ == "__main__":
    path = "./dataset/all-mtx/"

    filelist = []
    for file in glob.glob(os.path.join(path, "*.mtx")):
        filename = os.path.basename(file)
        filelist.append(filename[:-4])

    print(len(filelist))
    filelist = sorted(filelist, key=lambda x: x.lower())
    for mtx_name in filelist:
        print("-------{}-------".format(mtx_name))
        graph = graph_input(mtx_name=mtx_name, path=path)
        graph.load(load_from_mtx=True)
        # graph.reorder()
        graph.print_mtx_info()
        # bcsr_val = graph.bcsr_data.val
        # print("bcsr val shape: {}".format(bcsr_val.shape))
        # print(bcsr_val)
