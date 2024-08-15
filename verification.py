#!/usr/bin/env python3
import torch
import time
import os
import glob
from graph_input import graph_input
from tqdm import *


class SpMM_verification(object):
    def __init__(self, dim, rowPtr, colIdx, val, baseline=False, baseline_round=100):
        self.baseline = baseline
        self.baseline_round = baseline_round
        self.rowPtr = rowPtr
        self.colIdx = colIdx
        self.csr_val = val
        self.num_nodes = len(rowPtr) - 1
        self.s = torch.sparse_csr_tensor(
            self.rowPtr,
            self.colIdx,
            self.csr_val,
            size=(self.num_nodes, self.num_nodes),
        )
        # self.edge_index = edge_index
        self.test_embedding = dim
        self.X = torch.ones(self.num_nodes, self.test_embedding)
        self.result_ref = None

    def reference(self, dim, device="cuda"):
        """
        SpMM result on CPU.
        dim could be different from self.test_embedding
        """
        if device == "cpu":
            print("# Reference result on CPU")
        elif device == "cuda":
            print("# Reference result on GPU")
        else:
            assert False
        if dim == None:
            X = self.X.to(device)
        else:
            X = torch.ones(self.num_nodes, dim).to(device)
        s = self.s.to(device)
        result_ref = torch.sparse.mm(s, X)
        self.result_ref = result_ref.to("cpu")

    def torch_compute(self, device):
        """
        Compute SpMM (neighbor aggregation)
        result on GPU.
        """
        if device == "cpu":
            print("# torch Compute result on CPU")
        elif device == "cuda":
            print("# torch Compute result on GPU")
        else:
            assert False
        X = self.X.to(device)
        s = self.s.to(device)
        self.result = torch.sparse.mm(s, X)
        round = self.baseline_round
        if self.baseline:
            for _ in range(10):
                self.result = torch.sparse.mm(s, X)
            start = time.perf_counter()
            for _ in range(round):
                self.result = torch.sparse.mm(s, X)
            torch.cuda.synchronize()
            dur = (time.perf_counter() - start) / round
            gflops = 2 * self.rowPtr[-1] / 1e9 * self.test_embedding
            print(
                "dim: {} torch SpMM (ms): {:.3f} gflops: {:.0f}".format(
                    self.test_embedding, dur * 1e3, gflops / dur
                )
            )
            return self.result, int((gflops / dur).item())

    def compare(self, result, mtx_name):
        if self.result_ref is None or result is None:
            raise ValueError("MUST compute result and result reference (CPU) first!!")
        error = torch.abs(self.result_ref - result)
        threshold = 1e-4 * torch.abs(self.result_ref)
        mask = error > threshold

        if mask.any() == False:
            print("# Verification PASSED")
        else:
            print("# Verification FAILED in {}".format(mtx_name))
            # diff = torch.abs(result - self.result_ref)
            # indices = torch.nonzero(diff)
            indices = torch.nonzero(mask)
            for idx in indices:
                row, col = idx[0].item(), idx[1].item()
                value_ref = self.result_ref[row, col].item()
                value_result = result[row, col].item()
                print(
                    "Mismatch at position ({}, {}): Expected {}, Found {}".format(
                        row, col, value_ref, value_result
                    )
                )
            raise ValueError("Verification FAILED in {}".format(mtx_name))


if __name__ == "__main__":
    path = "./artifact_dataset/all-mtx/rabbit/"
    filelist = []
    for file in glob.glob(os.path.join(path, "*.mtx")):
        filename = os.path.basename(file)
        filelist.append(filename[:-4])
    filelist.sort()
    
    for mtx_name in filelist:
        print("-------------{}-------------".format(mtx_name))
        graph = graph_input(mtx_name=mtx_name, path=path)
        graph.load(load_from_mtx=True)
        dim = 32
        spmm_veri = SpMM_verification(
            dim,
            graph.csr_data.rowPtr,
            graph.csr_data.colIdx,
            graph.csr_data.val,
            baseline=True,
        )
        spmm_veri.reference()
        spmm_veri.torch_compute("cpu")
        spmm_veri.torch_compute("cuda")
        print("=======================================")
