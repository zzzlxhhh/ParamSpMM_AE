#!/usr/bin/env python3
import os
import csv
import glob
import math
import _pickle as pickle
from PCSR import *

"""
    use trained SpMMDecider to configure several PCSRs 
    dim_list: a list of dim for loading pickel files, 
                each rnd_tree is a separate model for a specific dimension
    
     
"""


class SpMMDecider(object):
    def __init__(self, pickle_path):
        if pickle_path == None:
            raise ValueError("pickle_path must be assigned")
        self.pickle_path = pickle_path
        self.tree_dict = {}
        self.model_dict = {}

    def load_pickle(self, dim):
        """
        load pickle file based on dim
        Args:
            dim (_type_): _description_
            pickle_path (_type_): _description_
        """
        if dim in self.tree_dict:
            return self.tree_dict[dim]
        pickle_file = self.pickle_path + "rnd_tree_" + str(dim) + ".pkl"
        if not os.path.exists(pickle_file):
            raise ValueError("pickle file not found for dim-" + str(dim) + " at path: " + pickle_file + "\n")
        else:
            self.tree_dict[dim] = pickle.load(open(pickle_file, "rb"))
            return self.tree_dict[dim]

    def gen_model_dict(self, dim_list):
        """
        load pickle files and generate model_dict
        """
        model_dict = {}
        if dim_list == None or len(dim_list) == 0:
            raise ValueError("dim_list must be assigned and non-empty")
        for dim in dim_list:
            # approx model
            # d = math.ceil(dim / 16) * 16
            if dim % 32 == 0:
                d = dim
            else:
                d = (math.ceil(dim / 32) * 2 - 1) * 16
            # if d is already in model_dict assign it to model[dim]
            if d in model_dict and d != dim:
                model_dict[dim] = model_dict[d]
            elif d not in model_dict:
                model_dict[dim] = self.load_pickle(d)
        self.model_dict = model_dict

    def decode_tree(self, code, dim, vec_size_list=[1, 2], warpNum_list=[4, 8]):
        """
        decode the output code of rnd_tree
        return <if_split,vec_size,CF,blkWarpNum>
        """
        if_split = code % 2
        CF_list = dynamic_CF_prompter(dim)
        # dcode vec_size
        len_vec_list = len(vec_size_list)
        vec_size = vec_size_list[code // 2 % len_vec_list]
        len_warp_list = len(warpNum_list)
        # index of CF
        len_CF_list = len(CF_list)
        m = code // (len_vec_list * 2) % len_CF_list
        CF = CF_list[m]
        m = code // (len_CF_list * len_vec_list * 2)
        blkWarpNum = warpNum_list[m]
        return if_split, vec_size, CF, blkWarpNum

    def config_PCSR(self, graph, dim):
        """
        configure PCSR based on graph and dim_list
        return a dict of PCSR
        """

        # load pickle file
        if dim not in self.model_dict:
            raise ValueError("dim-" + str(dim) + " not found in model_dict")
        decider = self.model_dict[dim]
        code = decider.predict(graph.get_info()).item()
        return self.decode_tree(code, math.ceil(dim / 16) * 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./dataset/all-mtx/",
        help="path to the mtx files",
    )
    parser.add_argument(
        "--dim_list",
        nargs="*",
        type=int,
        # , 144, 160, 176, 192, 208, 224, 240, 256
        default=[16, 32, 48, 64, 80, 96, 112, 128],
    )
    parser.add_argument("--mtx_name", type=str, default="", help="mtx file name")
    args = parser.parse_args()
    print(args)
    path = args.path
    dim_list = args.dim_list
    mtx_name = args.mtx_name

    pickle_path = "./models/"
    filelist = []
    if mtx_name == "":
        for file in glob.glob(os.path.join(path, "*.mtx")):
            filename = os.path.basename(file)
            filelist.append(filename[:-4])
        filelist.sort()
    else:
        filelist.append(mtx_name)
    SpMMDecider = SpMMDecider(pickle_path)
    SpMMDecider.gen_model_dict(dim_list)

    for mtx_name in filelist:
        graph = graph_input(mtx_name=mtx_name, path=path)
        print("-------------mtx: {}-------------".format(mtx_name))
        graph.load(load_from_mtx=True, if_mask=True)
        spmm_veri = SpMM_verification(
            1,
            graph.csr_data.rowPtr,
            graph.csr_data.colIdx,
            graph.csr_data.val,
            baseline=True,
        )
        coo_baseline = coo_spmm(graph)
        csc_baseline = csc_spmm(graph)
        paramSpMM_dict = {}
        for dim in dim_list:
            spmm_veri.reference(dim)
            # get config from SpMM-decider
            if_split, vec_size, CF, blkWarpNum = SpMMDecider.config_PCSR(graph, dim)
            paramSpMM = pcsr_spmm(
                graph=graph,
                vec_size=vec_size,
                if_split=if_split,
                CF=CF,
                blkWarpNum=blkWarpNum,
            )
            paramSpMM.get_spmm_result(dim)
            spmm_veri.compare(paramSpMM.result, mtx_name)
            paramSpMM_dict[dim] = paramSpMM
        for dim in dim_list:
            # ParamSpMM profiling
            param_tp = paramSpMM_dict[dim].profiling(dim)
            # baseline profiling
            Param10 = pcsr_spmm(
                graph,
                vec_size=1,
                if_split=False,
            )
            cootp = coo_baseline.profiling(dim)
            csctp = csc_baseline.profiling(dim)
            # cusparse
            cutp = Param10.cusparse_ref(dim, is_profiling=True)
            # GE-SpMM
            getp = Param10.gespmm_ref(dim, is_profiling=True)

            print("=======================================")
