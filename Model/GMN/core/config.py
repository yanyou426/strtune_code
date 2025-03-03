import json
import os

import logging
log = logging.getLogger('gnn')

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pickle,math
from numpy import dot
from numpy.linalg import norm
from transformers import RobertaTokenizerFast
from transformers import pipeline


def dump_config_to_json(config, outputdir):
    """
    Dump the configuration file to JSON

    Args:
        config: a dictionary with model configuration
        outputdir: path of the output directory
    """
    with open(os.path.join(outputdir, "config.json"), "w") as f_out:
        json.dump(config, f_out)
    return


def get_use_features(features_type):
    """Do not use features if the option is selected."""
    if features_type == "nofeatures":
        return False
    return True


def get_bb_features_size(features_type):
    """Return features size by type."""
    if features_type == "nofeatures":
        return 7
    if features_type == "opc":
        return 64
    if features_type == 'roberta':
        return 768
    raise ValueError("Invalid features_type")


def update_config_datasetone(config_dict, outputdir, featuresdir):
    """Config for Dataset-1."""
    inputdir = "~/workspace/binary/DBs/Dataset-1-new"

    # Training
    config_dict['training']['df_train_path'] = \
        os.path.join(inputdir, "all_training_Dataset-1_IR.csv")
    config_dict['training']['features_train_path'] = \
        os.path.join(
            featuresdir, "Dataset-1-new_training",
            "cfg_cdg_ddg_opc.json")

    # Validation
    valdir = os.path.join(inputdir, "pairs", "validation")
    config_dict['validation'] = dict(
        positive_path=os.path.join(valdir, "pos_validation_Dataset-1.csv"),
        negative_path=os.path.join(valdir, "neg_validation_Dataset-1.csv"),
        features_validation_path=os.path.join(
            featuresdir,
            "Dataset-1-new_validation",
            "cfg_cdg_ddg_opc.json")
    )

    # Testing
    testdir = os.path.join(inputdir, "pairs", "testing")
    config_dict['testing'] = dict(
        full_tests_inputs=[
            # "../../../DBs/Dataset-1-new/pairs/testing/neg_rank5_testing_Dataset-1.csv",
            "../../../DBs/Dataset-1-new/pairs/testing/neg_rank7_testing_Dataset-1.csv",
            # "../../../DBs/Dataset-1-new/pairs/testing/neg_rank7_testing_Dataset-1.csv",
            # "../../../DBs/Dataset-1-new/pairs/testing/neg_testing_Dataset-1.csv",
            # "../../../DBs/Dataset-1-new/pairs/testing/pos_rank5_testing_Dataset-1.csv",
            "../../../DBs/Dataset-1-new/pairs/testing/pos_rank7_testing_Dataset-1.csv",
            # "../../../DBs/Dataset-1-new/pairs/testing/pos_rank7_testing_Dataset-1.csv",
            # "../../../DBs/Dataset-1-new/pairs/testing/pos_testing_Dataset-1.csv"
        ],
        full_tests_outputs=[
            # os.path.join(outputdir, "neg_rank5_testing_Dataset-1_sim.csv"),
            os.path.join(outputdir, "neg_rank7_testing_Dataset-1_sim.csv"),
            # os.path.join(outputdir, "neg_rank7_testing_Dataset-1_sim.csv"),
            # os.path.join(outputdir, "neg_testing_Dataset-1_sim.csv"),
            # os.path.join(outputdir, "pos_rank5_testing_Dataset-1_sim.csv"),
            os.path.join(outputdir, "pos_rank7_testing_Dataset-1_sim.csv"),
            # os.path.join(outputdir, "pos_rank7_testing_Dataset-1_sim.csv"),
            # os.path.join(outputdir, "pos_testing_Dataset-1_sim.csv")
        ],
        features_testing_path=os.path.join(
            featuresdir,
            "Dataset-1-new_testing",
            "cfg_cdg_ddg_opc.json")
    )
    


def update_config_datasetfour(config_dict, outputdir, featuresdir):
    # """Config for Dataset."""
    inputdir = "~/workspace/binary/DBs/Dataset-4"
    in1 = "~/workspace/binary/DBs/Dataset-1-new"

    # Training
    config_dict['training']['df_train_path'] = \
        os.path.join(in1, "training_Dataset-1.csv")
    config_dict['training']['features_train_path'] = \
        os.path.join(
            featuresdir, "Dataset-1-new_training",
            "cfg_cdg_ddg_strand.json")

    # Validation
    valdir = os.path.join(in1, "pairs", "validation")
    config_dict['validation'] = dict(
        positive_path=os.path.join(valdir, "pos_validation_Dataset-1.csv"),
        negative_path=os.path.join(valdir, "neg_validation_Dataset-1.csv"),
        features_validation_path=os.path.join(
            featuresdir,
            "Dataset-1-new_validation",
            "cfg_cdg_ddg_strand.json")
    )


#     # Testing
    testdir = os.path.join(inputdir, "pairs", "testing")
    config_dict['testing'] = dict(
        full_tests_inputs=[
            os.path.join(testdir, "neg_rank1_testing_Dataset-4.csv"),
            os.path.join(testdir, "neg_testing_Dataset-4.csv"),
            os.path.join(testdir, "pos_rank1_testing_Dataset-4.csv"),
            os.path.join(testdir, "pos_testing_Dataset-4.csv")
        ],
        full_tests_outputs=[
            os.path.join(outputdir, "neg_rank_testing_Dataset-4_sim.csv"),
            os.path.join(outputdir, "neg_testing_Dataset-4_sim.csv"),
            os.path.join(outputdir, "pos_rank_testing_Dataset-4_sim.csv"),
            os.path.join(outputdir, "pos_testing_Dataset-4_sim.csv")
        ],
        features_testing_path=os.path.join(
            featuresdir,
            "Dataset-4_testing",
            "cfg_cdg_ddg_strand.json")
    )



def update_config_datasettwo(config_dict, outputdir, featuresdir):
    """Config for Dataset-2."""
    testdir = "/input/Dataset-2/pairs"
    config_dict['testing'] = dict(
        full_tests_inputs=[
            os.path.join(testdir, "neg_rank_testing_Dataset-2.csv"),
            os.path.join(testdir, "neg_testing_Dataset-2.csv"),
            os.path.join(testdir, "pos_rank_testing_Dataset-2.csv"),
            os.path.join(testdir, "pos_testing_Dataset-2.csv")
        ],
        full_tests_outputs=[
            os.path.join(outputdir, "neg_rank_testing_Dataset-2_sim.csv"),
            os.path.join(outputdir, "neg_testing_Dataset-2_sim.csv"),
            os.path.join(outputdir, "pos_rank_testing_Dataset-2_sim.csv"),
            os.path.join(outputdir, "pos_testing_Dataset-_sim2.csv")
        ],
        features_testing_path=os.path.join(
            featuresdir,
            "Dataset-2",
            "graph_func_dict_strand.json")
    )


def update_config_datasetvuln(config_dict, outputdir, featuresdir):
    """Config for Dataset-Vulnerability."""
    inputdir = "~/workspace/binary/DBs/Dataset-1-new"

    # Training
    config_dict['training']['df_train_path'] = \
        os.path.join(inputdir, "all_training_Dataset-1_IR.csv")
    config_dict['training']['features_train_path'] = \
        os.path.join(
            featuresdir, "Dataset-1-new_training",
            "cfg_ddg_opc.json")

    # Validation
    valdir = os.path.join(inputdir, "pairs", "validation")
    config_dict['validation'] = dict(
        positive_path=os.path.join(valdir, "pos_validation_Dataset-1.csv"),
        negative_path=os.path.join(valdir, "neg_validation_Dataset-1.csv"),
        features_validation_path=os.path.join(
            featuresdir,
            "Dataset-1-new_validation",
            "cfg_ddg_opc.json")
    )
    
    testdir = "~/workspace/binary/DBs/Dataset-Vul/pairs"
    config_dict['testing'] = dict(
        full_tests_inputs=[
            os.path.join(testdir, "pairs_testing_Dataset-Vulnerability.csv")
        ],
        full_tests_outputs=[
            os.path.join(outputdir, "pairs_testing_Dataset-Vulnerability.csv")
        ],
        features_testing_path=os.path.join(
            featuresdir,
            "Dataset-Vul_testing",
            "cfg_ddg_opc.json")
    )


def get_config(args):
    """The default configs."""
    NODE_STATE_DIM = 32
    GRAPH_REP_DIM = 128
    # EDGE_STATE_DIM = 8

    graph_embedding_net_config = dict(
        node_state_dim=NODE_STATE_DIM,
        edge_hidden_sizes=[NODE_STATE_DIM * 2, NODE_STATE_DIM * 2],
        node_hidden_sizes=[NODE_STATE_DIM * 2],
        n_prop_layers=10,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used
        # here.
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # *FS option
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can
        # help.
        layer_norm=False)

    graph_matching_net_config = graph_embedding_net_config.copy()

    # Alternatives are 'euclidean', 'dotproduct', 'cosine'
    graph_matching_net_config['similarity'] = 'dotproduct'

    config_dict = dict(
        encoder=dict(
            node_hidden_sizes=[NODE_STATE_DIM],
            edge_hidden_sizes=None),

        aggregator=dict(
            node_hidden_sizes=[GRAPH_REP_DIM],
            graph_transform_sizes=[GRAPH_REP_DIM],
            gated=True,
            aggregation_type='sum'),

        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,

        model_type=args.model_type,
        max_vertices=-1,
        edge_feature_dim=3,

        features_type=args.features_type,
        bb_features_size=get_bb_features_size(args.features_type),
        data=dict(
            use_features=get_use_features(args.features_type)),

        training=dict(
            mode=args.training_mode,
            # Alternative is 'hamming' ('margin' == -euclidean)
            loss='margin',
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in
            # the model we can add `snt.LayerNorm` to the outputs of each layer
            # , the aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            learning_rate=1e-3,
            num_epochs=args.num_epochs,
            print_after=100),
        validation=dict(),
        testing=dict(),

        batch_size=20,
        checkpoint_dir=args.checkpointdir,
        seed=11
    )
    
    if args.robertamodel:
        config_dict['bertmodel'] = SentenceTransformer(args.robertamodel)
    else:
        config_dict['bertmodel'] = None
        
    if args.dataset == 'one':
        update_config_datasetone(
            config_dict, args.outputdir, args.featuresdir)
    elif args.dataset == 'two':
        update_config_datasettwo(
            config_dict, args.outputdir, args.featuresdir)
    elif args.dataset == 'vuln':
        update_config_datasetvuln(
            config_dict, args.outputdir, args.featuresdir)
    elif args.dataset == 'four':
        update_config_datasetfour(
            config_dict, args.outputdir, args.featuresdir)

    return config_dict
