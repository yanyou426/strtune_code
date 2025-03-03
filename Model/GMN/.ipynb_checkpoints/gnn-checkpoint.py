import argparse
import coloredlogs
import logging
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import networkx as nx
from scipy.sparse import coo_matrix

from core import GNNModel
from core import dump_config_to_json
from core import get_config
import json
from core import str_to_scipy_sparse
import networkx as nx
import copy
log = None

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pickle,math
from numpy import dot
from numpy.linalg import norm
from transformers import RobertaTokenizerFast
from transformers import pipeline


def set_logger(debug, outputdir):
    """
    Set logger level, syntax, and logfile

    Args:
        debug: if True, set the log level to DEBUG
        outputdir: path of the output directory for the logfile
    """
    LOG_NAME = 'gnn'

    global log
    log = logging.getLogger(LOG_NAME)

    fh = logging.FileHandler(os.path.join(
        outputdir, '{}.log'.format(LOG_NAME)))
    fh.setLevel(logging.DEBUG)

    fmt = '%(asctime)s %(levelname)s:: %(message)s'
    formatter = coloredlogs.ColoredFormatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    if debug:
        loglevel = 'DEBUG'
    else:
        loglevel = 'INFO'
    coloredlogs.install(fmt=fmt,
                        datefmt='%H:%M:%S',
                        level=loglevel,
                        logger=log)
    return


def model_train(config, restore):
    """
    Train the model

    Args:
        config: model configuration dictionary
        restore: boolean. If True, continue the training from the latest
          checkpoint
    """
    gnn_model = GNNModel(config)
    gnn_model.model_train(restore)
    return


def model_validate(config):
    """
    Evaluate the model on validation dataset

    Args:
        config: model configuration dictionary
    """
    gnn_model = GNNModel(config)
    gnn_model.model_validate()
    return


def model_test(config):
    """
    Test the model

    Args:
        config: model configuration dictionary
    """
    gnn_model = GNNModel(config)
    gnn_model.model_test()
    return


def compare_list(graph_list, fea_1, fea_2, h=1, node_label=True):
        """Compute the all-pairs kernel values for a list of graphs.

        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)
        h : interger
            Number of iterations.
        node_label : boolean
            Whether to use original node labels. True for using node labels
            saved in the attribute 'node_label'. False for using the node
            degree of each node as node attribute.

        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.

        """
        graphs = graph_list
        n = len(graph_list)
        lists = [0] * n
        k = [0] * (h + 1)
        n_nodes = 0
        n_max = 0
        listt = [0] * n
        # Compute adjacency lists and n_nodes, the total number of nodes in the dataset.
        for i in range(n):
            lists[i] = list(graph_list[i].adjacency())
            # print(lists[i])
            n_nodes = n_nodes + graph_list[i].number_of_nodes()
            listt[i] = []
            for t in range(len(lists[i])):
                dic = lists[i][t][1]
                tmp_list = []
                if len(dic.keys()):
                    for key in dic.keys():
                        tmp_list.append(key)
                listt[i].append(tmp_list)
            # print(listt[i])
            # Computing the maximum number of nodes in the graphs. It
            # will be used in the computation of vectorial
            # representation.
            if(n_max < graph_list[i].number_of_nodes()):
                n_max = graph_list[i].number_of_nodes()

        phi = np.zeros((n_max, n), dtype=np.uint64)
        # print(phi.shape)
        # INITIALIZATION: initialize the nodes labels for each graph
        # with their labels or with degrees (for unlabeled graphs)

        labels = [0] * n
        label_lookup = {}
        label_counter = 0
        num2vec = dict()

        # label_lookup is an associative array, which will contain the
        # mapping from multiset labels (strings) to short labels
        # (integers)
        if node_label is True:
            for i in range(n):
                if i == 0:
                    l_aux = fea_1
                elif i == 1:
                    l_aux = fea_2
                # print(l_aux.shape)
                labels[i] = np.zeros(l_aux.shape[0], dtype=np.int32)
                # print(labels[i])
                for j in range(l_aux.shape[0]):
                    vec = l_aux[j, :]
                    # get the feature vector of each node
                    # print(vec)
                    found = 0
                    idx = 0
                    for key, value in num2vec.items():
                        if (value == vec).all():
                            found = 1
                            idx = key
                            break
                    if found == 0:
                        idx = len(num2vec.keys())
                        num2vec[idx] = vec
                    if not (idx in label_lookup):
                        label_lookup[idx] = label_counter
                        labels[i][j] = label_counter
                        label_counter += 1
                    else:
                        labels[i][j] = label_lookup[idx]
                    # labels are associated to a natural number
                    # starting with 0.
                    phi[labels[i][j], i] += 1 # the first idx should be used when phi is set to be (n_max * n)
        else:
            for i in range(n):
                labels[i] = np.array(list(graph_list[i].out_degree()))[:,1]
                # print(labels[i])
                for j in range(len(labels[i])):
                    phi[labels[i][j], i] += 1
                    
        
        # print(phi)
        # Simplified vectorial representation of graphs (just taking
        # the vectors before the kernel iterations), i.e., it is just
        # the original nodes degree.
        vectors = np.copy(phi.transpose())
#         # print(phi)
        k = np.dot(phi.transpose(), phi)
#         # print(k)
        
#         # MAIN LOOP
        it = 0
        new_labels = copy.deepcopy(labels)

        while it < h:
            # create an empty lookup table
            label_lookup = {}
            label_counter = 0

            phi = np.zeros((n_nodes, n), dtype=np.uint64)
            for i in range(n):
                # print(len(lists[i]))
                for v in range(len(listt[i])):
                    # form a multiset label of the node v of the i'th graph
                    # and convert it to a string
                    long_label = np.concatenate((np.array([labels[i][v]]), np.sort(labels[i][listt[i][v]])))
                    long_label_string = str(long_label)
                    # if the multiset label has not yet occurred, add it to the
                    # lookup table and assign a number to it
                    if not (long_label_string in label_lookup):
                        label_lookup[long_label_string] = label_counter
                        new_labels[i][v] = label_counter
                        label_counter += 1
                    else:
                        new_labels[i][v] = label_lookup[long_label_string]
                # fill the column for i'th graph in phi
                aux = np.bincount(new_labels[i].reshape((-1)))
                phi[new_labels[i], i] += aux[new_labels[i]].astype('uint64')

            k += np.dot(phi.transpose(), phi)
            labels = copy.deepcopy(new_labels)
            it = it + 1

#         # Compute the normalized version of the kernel
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])
        # print(k_norm)

        return k_norm

def compare(g_1, g_2, fea_1, fea_2, h=1, node_label=True):
    """Compute the kernel value (similarity) between two graphs.
    The kernel is normalized to [0,1] by the equation:
    k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2))
    """
    gl = [g_1, g_2]
    return compare_list(gl, fea_1, fea_2, h, node_label)[0, 1]

def test_wl():
    inputdir = "~/workspace/binary/DBs/Dataset-3"
    outputdir = "~/workspace/binary/Models/GGSNN-GMN/NeuralNetwork/Dataset-3_testing/wl"
    featuresdir = "~/workspace/binary/Models/GGSNN-GMN/Preprocessing"
    testdir = os.path.join(inputdir, "pairs", "testing")
    config_dict = dict(
        full_tests_inputs=[
            # os.path.join(testdir, "neg_rank_testing_Dataset-3.csv"),
            os.path.join(testdir, "neg_testing_Dataset-3.csv"),
            os.path.join(testdir, "pos_rank_testing_Dataset-3.csv"),
            os.path.join(testdir, "pos_testing_Dataset-3.csv")
        ],
        full_tests_outputs=[
            # os.path.join(outputdir, "neg_rank_testing_Dataset-3_sim.csv"),
            os.path.join(outputdir, "neg_testing_Dataset-3_sim.csv"),
            os.path.join(outputdir, "pos_rank_testing_Dataset-3_sim.csv"),
            os.path.join(outputdir, "pos_testing_Dataset-3_sim.csv")
        ],
        features_testing_path=os.path.join(
            featuresdir,
            "Dataset-3_testing",
            "graph_func_dict_strand.json")
    )
    with open('../Preprocessing/Dataset-3_testing/graph_func_dict_strand.json') as f:
        info = json.load(f)
    # Evaluate the full testing dataset
    for df_input_path, df_output_path in \
        zip(config_dict['full_tests_inputs'],
            config_dict['full_tests_outputs']):

        df = pd.read_csv(df_input_path, index_col=0) # df: num * 7 cols
        list_length = df.shape[0]
        similarity_list = list()
        for idx in range(list_length):
            f1_path = df.iloc[idx][0]
            f1_fva = df.iloc[idx][1]
            f1_info = info[f1_path][f1_fva]
            # print(f1_info)
            g_pos = nx.DiGraph(str_to_scipy_sparse(f1_info['graph']))
            f_pos = str_to_scipy_sparse(f1_info['opc'])
            f2_path = df.iloc[idx][3]
            f2_fva = df.iloc[idx][4]
            f2_info = info[f2_path][f2_fva]
            # print(f2_info)
            g_neg = nx.DiGraph(str_to_scipy_sparse(f2_info['graph']))
            f_neg = str_to_scipy_sparse(f2_info['opc'])
            
            sim = compare(g_pos, g_neg, f_pos, f_neg, h=20, node_label=False)
            similarity_list.append(sim)
        # print(similarity_list)
        # print(len(similarity_list))

        # Save the cosine similarity
        df['sim'] = similarity_list[:df.shape[0]]

        # Save the result to CSV
        df.to_csv(df_output_path)
        print("Result CSV saved to {}".format(df_output_path))

            

def main():
    parser = argparse.ArgumentParser(
        prog='gnn',
        description='GGSNN and GMN models',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--debug', action='store_true',
                        help='Log level debug')

    group0 = parser.add_mutually_exclusive_group(required=True)
    group0.add_argument('--train', action='store_true',
                        help='Train the model')
    group0.add_argument('--validate', action='store_true',
                        help='Run model validation')
    group0.add_argument('--test', action='store_true',
                        help='Run model testing')

    parser.add_argument("--featuresdir", required = True,
                        help="Path to the Preprocessing dir")

    parser.add_argument("--features_type", default="opc",
                        choices=["nofeatures",
                                 "opc"],
                        help="Select the type of BB features")

    parser.add_argument("--model_type", default="matching",
                        choices=["embedding", "matching"],
                        help="Select the type of network (default: matching)")

    parser.add_argument("--training_mode", default="pair",
                        choices=["pair", "triplet"],
                        help="Select the type of network")

    parser.add_argument('--num_epochs', type=int,
                        required=False, default=2,
                        help='Number of training epochs')

    parser.add_argument('--restore',
                        action='store_true', default=False,
                        help='Continue the training from the last checkpoint')

    parser.add_argument('--dataset', default="one",
                        choices=['one', 'two', 'vuln', 'zlib', 'three', 'four'],
                        help='Choose the dataset to use for the train or test')

    parser.add_argument('-c', '--checkpointdir', required=True,
                        help='Input/output for model checkpoint')

    parser.add_argument('-o', '--outputdir', required=True,
                        help='Output dir')
    
    parser.add_argument('-b', '--robertamodel', required=True,
                        help='robertamodel dir')
    
    # parser.add_argument('-a', '--algorithm', required=True,
    #                     choices=['GNN', 'WL'], help='Choose the WL or nn to calculate similarity')
    
    args = parser.parse_args()
    
    # if args.algorithm == 'GNN':
    # Create the output directory
    if args.outputdir:
        if not os.path.isdir(args.outputdir):
            os.mkdir(args.outputdir)
            print("Created outputdir: {}".format(args.outputdir))

    if args.featuresdir:
        if not os.path.isdir(args.featuresdir):
            print("[!] Non existing directory: {}".format(args.featuresdir))
            return

    if args.checkpointdir:
        if not os.path.isdir(args.checkpointdir):
            os.mkdir(args.checkpointdir)
            print("Created checkpointdir: {}".format(args.checkpointdir))

    # Create logger
    set_logger(args.debug, args.outputdir)

    # Load the model configuration and save to file
    config = get_config(args)
    config_w = copy.deepcopy(config)
    config_w.pop('bertmodel')
    dump_config_to_json(config_w, args.outputdir)

    if args.train:
        log.info("Running model training")
        model_train(config, restore=args.restore)

    if args.validate:
        log.info("Running model validation")
        model_validate(config)

    if args.test:
        log.info("Running model testing")
        model_test(config)
    
        

    return


if __name__ == '__main__':
    main()
