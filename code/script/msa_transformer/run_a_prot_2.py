#/usr/bin/env python3

#
# Arontier Inc.: Artificial Intelligence in Precision Medicine
# Copyright: 2018-present
#

import os
import torch
#from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import esm
import json
import numpy as np
import argparse
from einops import rearrange
import string
from collections import defaultdict 
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from script.msa_transformer import aln_to_a3m

DIST_CUT_OFF = 8  # 8A 10A 12A
DIST_BIN_NUM = 37
OMEGA_BIN_NUM = 25
THETA_BIN_NUM = 25
PHI_BIN_NUM = 13
MATH_PI = np.pi
DIST_BOUNDARY = np.linspace(2, 20, num=DIST_BIN_NUM)
OMEGA_BOUNDARY = np.linspace(-MATH_PI, MATH_PI, num=OMEGA_BIN_NUM)
THETA_BOUNDARY = np.linspace(-MATH_PI, MATH_PI, num=THETA_BIN_NUM)
PHI_BOUNDARY = np.linspace(0, MATH_PI, num=PHI_BIN_NUM)

MAX_TOKEN_NUM = 2 ** 14
MAX_MSA_ROW_NUM = 800
MIN_MSA_ROW_NUM = 16
MAX_MSA_COL_NUM = 1024

CONV_NET_FILTERS = 64
CONV_NET_KERNEL = 3
CONV_NET_LAYERS = 28
torch.set_grad_enabled(False)


def elu():
    return nn.ELU(inplace=True)
def relu():
    return nn.ReLU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)


def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)


class trRosettaNetwork(nn.Module):
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        self.linear_proj = nn.Sequential(
            nn.Linear(768, 384),
            nn.InstanceNorm1d(384),
            relu(),
            nn.Linear(384, 192),
            nn.InstanceNorm1d(192),
            relu(),
            nn.Linear(192, 128),
        )

        self.first_block = nn.Sequential(
            conv2d(400, 256, 1),
            instance_norm(256),
            relu(),
            conv2d(256, 128, 1),
            instance_norm(128),
            relu(),
            conv2d(128, filters, 1),
            instance_norm(filters),
            relu(),
        )

        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4]

        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            instance_norm(filters),
            relu(),
            conv2d(filters, filters, kernel, dilation=dilation),
            nn.Dropout(p=0.15),
            relu(),
            conv2d(filters, filters, kernel, dilation=dilation),
        ) for dilation in dilations])

        self.to_distance = conv2d(filters, DIST_BIN_NUM, 1)
        self.to_omega = conv2d(filters, OMEGA_BIN_NUM, 1)
        self.to_theta = conv2d(filters, THETA_BIN_NUM, 1)
        self.to_phi = conv2d(filters, PHI_BIN_NUM, 1)

    def forward(self, msa_query_embeddings, msa_row_attentions):

        msa_query_embeddings = self.linear_proj(msa_query_embeddings)
        msa_query_embeddings = msa_query_embeddings.permute((0, 2, 1))
        msa_query_embeddings_row_expand = msa_query_embeddings.unsqueeze(2).repeat(1, 1, msa_query_embeddings.shape[-1],1)

        msa_query_embeddings_col_expand = msa_query_embeddings.unsqueeze(3).repeat(1, 1, 1, msa_query_embeddings.shape[-1])

        msa_query_embeddings_out_concat = torch.cat([msa_query_embeddings_row_expand, msa_query_embeddings_col_expand], dim=1)

        msa_row_attentions = rearrange(msa_row_attentions, 'b l h i j -> b (l h) i j')

        msa_row_attentions_symmetrized = 0.5 * (msa_row_attentions + msa_row_attentions.permute((0, 1, 3, 2)))
        conv_input = torch.cat([msa_query_embeddings_out_concat, msa_row_attentions_symmetrized], dim=1)

        x = self.first_block(conv_input)
        #row_conv=self.second_block(msa_row_attentions_symmetrized)
        #for layer in self.layers:
        #    x = x + layer(x)


        #prob_theta = self.to_theta(x)  # anglegrams for theta
        #prob_phi = self.to_phi(x)  # anglegrams for phi

        #x = 0.5 * (x + x.permute((0, 1, 3, 2)))  # symmetrize
        #prob_distance = self.to_distance(x)  # distograms
        #prob_omega = self.to_omega(x)

        #return prob_distance, prob_omega, prob_theta, prob_phi
        
        return x,msa_row_attentions_symmetrized


# Define the value of 'c' based on your input size
c = 64  # Adjust this value based on your input size

# Create an instance of the model
# Print the model architecture

def read_msa_json(msa_json_path, msa_method, msa_row_num):

    with open(msa_json_path) as json_file:
        msa_coord_json_dict = json.load(json_file)

    if not msa_method:
        msa_method = list(msa_coord_json_dict['MSA'].keys())[0]

    msa_seq = msa_coord_json_dict['MSA'][msa_method]['sequences']
    msa_seq = [seq[0:2] for seq in msa_seq]
    query_seq = msa_seq[0][1]

    if msa_row_num > MAX_MSA_ROW_NUM:
        msa_row_num = MAX_MSA_ROW_NUM
        print(f"The MSA row num is larger than {MAX_MSA_ROW_NUM}. This program force the msa row to under {MAX_MSA_ROW_NUM}")

    msa_seq = msa_seq[: msa_row_num]

    return msa_seq, query_seq

def read_msa_file(filepath):

    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    with open(filepath,"r") as f:
        lines = f.readlines()
    # read file line by line
    for i in range(0,len(lines),2):

        seq = []
        seq.append(lines[i])
        seq.append(lines[i+1].rstrip().translate(table))
        seqs.append(seq)
    msa_row_num=int(len(lines)/2)
    #print(msa_row_num)
    if msa_row_num > MAX_MSA_ROW_NUM:
        msa_row_num = MAX_MSA_ROW_NUM
        print(f"The MSA row num is larger than {MAX_MSA_ROW_NUM}. This program force the msa row to under {MAX_MSA_ROW_NUM}")

    seqs = seqs[: msa_row_num]
    return seqs, seqs[0]

def extract_msa_transformer_features(msa_seq, msa_transformer, msa_batch_converter, device=torch.device("cpu")):
    msa_seq_label, msa_seq_str, msa_seq_token = msa_batch_converter([msa_seq])
    msa_seq_token = msa_seq_token.to(device)
    msa_row, msa_col = msa_seq_token.shape[1], msa_seq_token.shape[2]
    print(f"{msa_seq_label[0][0]}, msa_row: {msa_row}, msa_col: {msa_col}")

    if msa_col > MAX_MSA_COL_NUM:
        print(f"msa col num should less than {MAX_MSA_COL_NUM}. This program force the msa col to under {MAX_MSA_COL_NUM}")
    msa_seq_token = msa_seq_token[:, :, :MAX_MSA_COL_NUM]

    ### keys: ['logits', 'representations', 'col_attentions', 'row_attentions', 'contacts']
    msa_transformer_outputs = msa_transformer(
        msa_seq_token, repr_layers=[12],
        need_head_weights=True, return_contacts=True)
    msa_row_attentions = msa_transformer_outputs['row_attentions']
    msa_representations = msa_transformer_outputs['representations'][12]
    
    msa_query_representation = msa_representations[:, 0, 1:, :]  # remove start token
    #msa_representations=msa_representations[:,:,1:,:]
    msa_row_attentions = msa_row_attentions[..., 1:, 1:]  # remove start token
    #msa_col_attentions = msa_col_attentions[:,:,:,0,1:,1:]
    #print(msa_col_attnetions.size())

    return msa_query_representation, msa_row_attentions,1,1

import subprocess


#if __name__ == '__main__':
def calc_attention(data_dir,protein):

    parser = argparse.ArgumentParser(description='A Prot: input msa and output .npz for trrosetta structure modeling')
    parser.add_argument('-i', '--input_path', type=str, default='T0998.json',
                        help='input msa path')
    parser.add_argument('-o', '--output_path', type=str, default='T0998.npz',
                        help='output trrosetta network feature path')
    parser.add_argument('--conv_model_path', type=str,
                        default='script/msa_transformer/a_prot_resnet_weights.pth',
                        help='resnet model weight path')

    msa_args = parser.add_argument_group('MSA')

    msa_args.add_argument('--msa_method', type=str, help='input msa method')
    msa_args.add_argument('--msa_row_num', type=int, default=256,
                          help='input msa row num to msa transformer')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='choose device: cpu or gpu')


    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")

    ## if already have the msa_transformer_weight
    ## msa_transformer, msa_alphabet = esm.pretrained.load_model_and_alphabet_local(msa_transformer_weight_path)

    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()

    msa_transformer.to(device)
    msa_transformer.eval()

    for param in msa_transformer.parameters():
        param.requires_grad = False


    conv_model = trRosettaNetwork(filters=CONV_NET_FILTERS, kernel=CONV_NET_KERNEL, num_layers=CONV_NET_LAYERS)
    conv_model = conv_model.to(device)
    #    ch = torch.load(args.conv_model_path)
    if device.type == 'cpu':
        ch = torch.load(args.conv_model_path, map_location=torch.device('cpu'))
    else:
        ch = torch.load(args.conv_model_path)

    conv_model.load_state_dict(ch['conv_model'])
    conv_model.to(device)
    conv_model.eval()


    tool='PSI'    
    a3m_dir='%s/%s_modify.a3m'%(data_dir,protein)

    if not os.path.isfile(a3m_dir):
        aln_to_a3m.make_a3m_file(data_dir,protein)
        aln_to_a3m.make_a3m_modify_file(data_dir,protein)

    result_file=defaultdict(list)

    if os.path.isfile(a3m_dir):
        master_df=pd.DataFrame()
        msa_seq, query_seq = read_msa_file(a3m_dir)

        msa_row_num = len(msa_seq)
        msa_col_num = len(query_seq[1])
                    
        msa_query_representation, msa_row_attentions,msa_col_attentions, msa_representations = extract_msa_transformer_features(msa_seq,
                                                                                                    msa_transformer,
                                                                                                    msa_batch_converter,
                                                                                                    device=device)

        msa_query_representation.to(device)
        msa_row_attentions.to(device)
        print('msa query size is %s',msa_query_representation.shape)
        print('msa row attentions size is %s',msa_row_attentions.shape)
        x,row_attention=conv_model(msa_query_representation,msa_row_attentions)
        x2= x.permute((0, 1,3, 2))
        numpy_x=x.numpy()
        numpy_x2=x2.numpy()

        np.save('%s/%s_transformer_feature.npy'%(data_dir,protein), numpy_x)
        np.save('%s/%s_transformer_feature_trans.npy'%(data_dir,protein),numpy_x2)


    print("===================================")
    print("Done")
    print("===================================")
