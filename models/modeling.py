# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm
from torch.nn.modules.utils import _pair


import models.configs as configs

import numpy as np
import scipy


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings3D(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, num_frame, in_channels=3, temporal = True):
        super(Embeddings3D, self).__init__()
        
        self.temporal = temporal
        
        img_size = _pair(img_size)

        patch_size = config.patches["size"]
        
        assert img_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
                
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (num_frame // patch_size[2])
        
        if not self.temporal:
            self.tubelet_embedding = nn.Conv3d(in_channels=in_channels,
                                            out_channels=config.hidden_size,
                                            kernel_size=patch_size,
                                            stride=patch_size)
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+2, config.hidden_size))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        B = x.shape[0]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if not self.temporal:
            x = self.tubelet_embedding(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
        
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, num_frames, temporal, vis = False):
        super(Transformer, self).__init__()
        self.embedding = Embeddings3D(config, img_size=img_size,num_frame=num_frames, temporal=temporal)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        input_ids = self.embedding(input_ids)
        encoded, _ = self.encoder(input_ids)
        return encoded



class MyViViT(nn.Module):
    def __init__(self, config, image_size=224, num_classes=10, num_frames = 32,  pool = 'cls'):
        super().__init__()
        
        self.image_size = image_size
        self.hidden_dim = config.hidden_size     
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.label_smoothing = config.label_smoothing
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.spatial_transformer = Transformer(config.spatial,image_size,num_frames,temporal=False) 
        self.temporal_transformer = Transformer(config.temporal,image_size,num_frames,temporal = True)       
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x, labels=None):
        
        x = x.permute(0,2,3,4,1)        
        x = self.spatial_transformer(x)      
        x = self.temporal_transformer(x)        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]        
        logits = self.mlp_head(x)  

        if labels is not None:            
            labels_one_hot = F.one_hot(torch.tensor(labels), num_classes=self.num_classes).view(-1, self.num_classes)
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits.view(-1, self.num_classes), labels_one_hot)
            return loss      
        else:            
            return logits

    def prepare_conv3d_weight(self, weights, method):
        # Prepare kernel weights
        kernel_w = np2th(weights["embedding/kernel"],conv=True)
        expanding_dim = self.spatial_transformer.embedding.tubelet_embedding.weight.shape[4]
        # Prepare bias weights
        bias_w = np2th(weights["embedding/bias"])
        
        if method == "central_frame_initializer":
            init_index = expanding_dim // 2
            pad_w = torch.zeros(self.spatial_transformer.embedding.tubelet_embedding.weight.shape)
            pad_w[:,:,:,:,init_index] = kernel_w
            kernel_w = pad_w
        else:
            kernel_w = torch.unsqueeze(kernel_w, 4)
            kernel_w = kernel_w.expand(-1, -1, -1, -1, expanding_dim)
            kernel_w = torch.div(kernel_w, expanding_dim) 
            
    
        self.spatial_transformer.embedding.tubelet_embedding.weight.copy_(kernel_w)
        self.spatial_transformer.embedding.tubelet_embedding.bias.copy_(bias_w)
        
        return
    
    def load_temporal_positional_encoding(self, restored_posemb_old,n_tokens):
        restored_posemb = np.squeeze(restored_posemb_old)
        zoom = (n_tokens / restored_posemb.shape[0], 1)
        restored_posemb = scipy.ndimage.zoom(restored_posemb, zoom, order=1)
        restored_posemb = np.expand_dims(restored_posemb, axis=0)
        return restored_posemb
        
    def load_from(self, weights):
        with torch.no_grad():
            
            self.prepare_conv3d_weight(weights, method = "central_frame_initializer")
            self.spatial_transformer.embedding.cls_token.copy_(np2th(weights["cls"]))
            self.temporal_transformer.embedding.cls_token.copy_(np2th(weights["cls"]))
            
            self.spatial_transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.spatial_transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            
            self.temporal_transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.temporal_transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_temporal = self.load_temporal_positional_encoding(posemb, self.temporal_transformer.embedding.position_embeddings.shape[1])
            
            posemb_spatial = self.spatial_transformer.embedding.position_embeddings
            if posemb.size() != posemb_spatial.size():
                expand_factor = (posemb_spatial.shape[1]-1)//(posemb.shape[1]-1)
                cls_token_weight = torch.unsqueeze(posemb[:,0,:],0)
                tubelet_weight = posemb[:,1:,:].repeat(1,expand_factor,1)
                posemb = torch.cat((cls_token_weight,tubelet_weight),dim=1)

            self.spatial_transformer.embedding.position_embeddings.copy_(posemb)
            self.temporal_transformer.embedding.position_embeddings.copy_(np2th(posemb_temporal))   

            
            for bname, block in self.spatial_transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
                    
            for bname, block in self.temporal_transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

CONFIGS = {
    'ViViT-B/16x2': configs.get_vb16_config(),
    'ViViT-B/16x2-small': configs.get_vb16_config_small()
}
