#/usr/bin/env python3
import os
import torch
import torch.nn as nn
import esm
import json
import numpy as np
import argparse
from einops import rearrange
import string
from collections import defaultdict 
import pandas as pd 
from . import aln_to_a3m
import warnings

# InstanceNorm warning
warnings.filterwarnings(
    "ignore",
    message="input's size at dim=1 does not match num_features",
    category=UserWarning,
    module="torch.nn.modules.instancenorm"
)

# --- Constants ---
MAX_MSA_ROW_NUM = 800
MAX_MSA_COL_NUM = 1024
CONV_NET_FILTERS = 64
CONV_NET_KERNEL = 3
CONV_NET_LAYERS = 28

DIST_BIN_NUM = 37
OMEGA_BIN_NUM = 25
THETA_BIN_NUM = 25
PHI_BIN_NUM = 13
torch.set_grad_enabled(False)

# --- Basic layers ---
def relu():
    return nn.ReLU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)

# --- Model ---
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
        return x,msa_row_attentions_symmetrized


# --- IO helpers ---
def read_msa_file(filepath):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    with open(filepath,"r") as f:
        lines = f.readlines()
    for i in range(0,len(lines),2):
        seq = [lines[i], lines[i+1].rstrip().translate(table)]
        seqs.append(seq)
    msa_row_num=int(len(lines)/2)
    if msa_row_num > MAX_MSA_ROW_NUM:
        msa_row_num = MAX_MSA_ROW_NUM
        print(f"The MSA row num is larger than {MAX_MSA_ROW_NUM}, truncated.")
    return seqs[: msa_row_num], seqs[0]

def extract_msa_transformer_features(msa_seq, msa_transformer, msa_batch_converter, device=torch.device("cpu")):
    msa_seq_label, msa_seq_str, msa_seq_token = msa_batch_converter([msa_seq])
    msa_seq_token = msa_seq_token.to(device)
    msa_seq_token = msa_seq_token[:, :, :MAX_MSA_COL_NUM]

    outputs = msa_transformer(msa_seq_token, repr_layers=[12], need_head_weights=True, return_contacts=True)
    msa_row_attentions = outputs['row_attentions']
    msa_representations = outputs['representations'][12]

    msa_query_representation = msa_representations[:, 0, 1:, :]  # remove start token
    msa_row_attentions = msa_row_attentions[..., 1:, 1:]  # remove start token
    return msa_query_representation, msa_row_attentions

# --- Main function ---
def calc_attention(data_dir, enzyme):
    #parser = argparse.ArgumentParser(description='A Prot: input msa and output .npz for trrosetta structure modeling')
    #parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='choose device: cpu or gpu')
    #args = parser.parse_args()

    #device = torch.device("cpu") if args.device == 'cpu' or not torch.cuda.is_available() else torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()
    msa_transformer.to(device).eval()

    for param in msa_transformer.parameters():
        param.requires_grad = False

    conv_model = trRosettaNetwork(filters=CONV_NET_FILTERS, kernel=CONV_NET_KERNEL, num_layers=CONV_NET_LAYERS).to(device).eval()
    ch = torch.load("script/msa_transformer/a_prot_resnet_weights.pth", map_location=device)
    conv_model.load_state_dict(ch['conv_model'])

    a3m_dir=f"{data_dir}/{enzyme}_modify.a3m"
    if not os.path.isfile(a3m_dir):
        aln_to_a3m.make_a3m_file(data_dir,enzyme)
        aln_to_a3m.make_a3m_modify_file(data_dir,enzyme)

    msa_seq, _= read_msa_file(a3m_dir)
    msa_query_representation, msa_row_attentions = extract_msa_transformer_features(msa_seq, msa_transformer, msa_batch_converter, device=device)

    msa_query_representation.to(device)
    msa_row_attentions.to(device)

    x,_ = conv_model(msa_query_representation, msa_row_attentions)
    x_trans = x.permute((0, 1,3, 2))

    np.save(f"{data_dir}/{enzyme}_transformer_feature.npy", x.cpu().numpy())
    np.save(f"{data_dir}/{enzyme}_transformer_feature_trans.npy", x_trans.cpu().numpy())

    return x, x_trans