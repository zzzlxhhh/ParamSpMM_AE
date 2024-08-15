#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from tqdm import *
from scipy.sparse import *
from gnn_conv import *
from graph_input import *
from decider import *

# 16, 32, 64, 96, 128
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=str, choices=["True", "False"], default="False")
parser.add_argument("--hidden_list", nargs="*", type=int, default=[32], help="hidden size list")
parser.add_argument("--num_classes", type=int, default=32)
parser.add_argument("--num_features", type=int, default=32)
parser.add_argument("--num_epoches", type=int, default=200)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument(
    "--path", type=str, default="/mnt/data1/zhanglx/artifact_dataset/OGB/rabbit/"
)  # path of mtx file ./OGB/rabbit/
parser.add_argument("--pickle_path", type=str, default="./models/")  # path of pickle file
parser.add_argument("--mtx_name", type=str, default="")  # mtx name
parser.add_argument("--net", type=str, choices=["GCN", "GIN", "AGNN"], default="AGNN")

args = parser.parse_args()
print(args)
verbose = args.verbose == "True"
hidden_list = args.hidden_list
num_classes = args.num_classes
num_features = args.num_features
num_epoches = args.num_epoches
num_layers = args.num_layers
path = args.path
pickle_path = args.pickle_path
mtx_name = args.mtx_name
net = args.net


class Net_5layers(torch.nn.Module):
    def __init__(self, num_classes, num_features, x, spmm_dict, hidden):
        super(Net_5layers, self).__init__()
        self.spmm_dict = spmm_dict
        self.x = x  # feature embedding
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
        self.conv5 = GCNConv(hidden, num_classes)

    def forward(self):
        x = self.x
        x = F.relu(self.conv1(x, self.spmm_dict))
        x = F.relu(self.conv2(x, self.spmm_dict))
        x = F.relu(self.conv3(x, self.spmm_dict))
        x = F.relu(self.conv4(x, self.spmm_dict))
        x = self.conv5(x, self.spmm_dict)
        return F.log_softmax(x, dim=1)


class Net_3layers(torch.nn.Module):
    def __init__(self, num_classes, num_features, x, spmm_dict, hidden):
        super(Net_3layers, self).__init__()
        self.spmm_dict = spmm_dict
        self.x = x  # feature embedding
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, num_classes)

    def forward(self):
        x = self.x
        x = F.relu(self.conv1(x, self.spmm_dict))
        x = F.relu(self.conv2(x, self.spmm_dict))
        x = self.conv3(x, self.spmm_dict)
        return F.log_softmax(x, dim=1)


class AGNN(torch.nn.Module):
    def __init__(self, num_classes, num_features, x, spmm_dict, hidden, colIdx):
        super(AGNN, self).__init__()
        self.x = x
        self.spmm_dict = spmm_dict
        self.colIdx = colIdx
        self.conv1 = AGNNConv(num_features, hidden)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(AGNNConv(hidden, hidden))
        self.conv2 = AGNNConv(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self):
        x = self.x
        x = self.relu(self.conv1(x, self.spmm_dict, self.colIdx))
        x = F.dropout(x, training=self.training)
        for Gconv in self.hidden_layers:
            x = Gconv(x, self.spmm_dict, self.colIdx)
            x = self.relu(x)
        x = self.conv2(x, self.spmm_dict, self.colIdx)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, num_classes, num_features, x, spmm_dict, hidden):
        super(GIN, self).__init__()
        self.spmm_dict = spmm_dict
        self.x = x
        self.conv1 = GINConv(num_features, hidden)
        self.conv2 = GINConv(hidden, hidden)
        self.conv3 = GINConv(hidden, hidden)
        self.conv4 = GINConv(hidden, hidden)
        self.conv5 = GINConv(hidden, num_classes)

    def forward(self):
        x = self.x
        x = F.relu(self.conv1(x, self.spmm_dict))
        x = F.relu(self.conv2(x, self.spmm_dict))
        x = F.relu(self.conv3(x, self.spmm_dict))
        x = F.relu(self.conv4(x, self.spmm_dict))
        x = self.conv5(x, self.spmm_dict)
        return F.log_softmax(x, dim=1)


def train_profiling(model, num_epoches, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, capturable=True)
    optimizer.zero_grad()

    if verbose == False:
        # warm up
        for _ in range(1):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model()[:], graph.y[:])
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    start_train = time.perf_counter()
    for _ in tqdm(range(num_epoches)):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model()[:], graph.y[:])
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train
    # print("model_name:{} Time (ms): {:.3f}".format(model.spmm_dict, train_time * 1e3 / num_epoches))
    print("training time per epoch: {:.3f} (ms)".format(train_time * 1e3 / num_epoches))


if __name__ == "__main__":

    filelist = []
    if mtx_name == "":
        for file in glob.glob(os.path.join(path, "*.mtx")):
            filename = os.path.basename(file)
            filelist.append(filename[:-4])
        filelist.sort()
    else:
        filelist.append(mtx_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print("filelist: ", filelist[1])
    # filelist.append("ogbl-collab_rabbit")
    # filelist.append("ogbl-vessel_rabbit")
    # num_features = 128
    # num_classes = 16
    # hidden_list = [32, 64, 96, 128]

    # init
    SpMMDecider = SpMMDecider(pickle_path)
    SpMMDecider.gen_model_dict([16, 32, 48, 64, 80, 96, 112, 128])

    for hidden_size in hidden_list:
        for mtx_name in filelist:
            # dim_list for paramSpMM config
            dim_list = [hidden_size, num_classes]
            print("-------------mtx: {} hidden_size: {}-------------".format(mtx_name, hidden_size))
            assert torch.cuda.is_available()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            graph = graph_input(
                mtx_name=mtx_name,
                path=path,
            )

            # end2end training time of whole process
            start = time.perf_counter()
            if net == "GIN":
                graph.load(load_from_mtx=True, GIN_epsilon=0.5)
            elif net == "AGNN":
                graph.load(load_from_mtx=True, is_normalize=True, if_mask=True)
            elif net == "GCN":
                graph.load(load_from_mtx=True, is_normalize=True)
            graph.init_feat_embedding(num_features)
            graph.init_labels(num_classes)
            start_MLpre = time.perf_counter()
            paramSpMM_dict = {}
            for dim in dim_list:
                if_split, vec_size, CF, blkWarpNum = SpMMDecider.config_PCSR(graph, dim)
                paramSpMM = pcsr_spmm(
                    graph=graph,
                    vec_size=vec_size,
                    if_split=if_split,
                    CF=CF,
                    blkWarpNum=blkWarpNum,
                )
                paramSpMM_dict[dim] = paramSpMM
            print("SpMM-decider config time: {:.3f} (ms)".format((time.perf_counter() - start_MLpre) * 1e3))

            # print("colIdx: ", graph.csr_data.colIdx)
            if net == "GCN":
                model = Net_5layers(
                    graph.num_classes,
                    graph.num_features,
                    x=graph.x,
                    spmm_dict=paramSpMM_dict,
                    hidden=hidden_size,
                ).to(device)
            elif net == "AGNN":
                colIdx = graph.csr_data.colIdx.to(device)
                colIdx = colIdx.unsqueeze(-1)
                # padding
                print("mtx_node", graph.mtx_nodes)
                graph.init_feat_embedding(num_features, 2)
                graph.init_labels(num_classes, 2)
                model = AGNN(
                    graph.num_classes,
                    graph.num_features,
                    x=graph.x,
                    spmm_dict=paramSpMM_dict,
                    hidden=hidden_size,
                    colIdx=colIdx,
                ).to(device)
            elif net == "GIN":
                graph.init_feat_embedding(num_features, 2)
                graph.init_labels(num_classes, 2)
                model = GIN(
                    graph.num_classes,
                    graph.num_features,
                    x=graph.x,
                    spmm_dict=paramSpMM_dict,
                    hidden=hidden_size,
                ).to(device)

            # model = Net_3layers(
            #     graph.num_classes,
            #     graph.num_features,
            #     x=graph.x,
            #     spmm_dict=paramSpMM_dict,
            #     hidden=hidden_size,
            # ).to(device)
            torch.cuda.synchronize()
            train_profiling(model, num_epoches, verbose=verbose)
            end2end_time = time.perf_counter() - start
            print("end2end time: {:.3f} (ms)".format(end2end_time * 1e3))
