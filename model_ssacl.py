# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MRM: https://github.com/RL4M/MRM-pytorch
# MaCo: https://github.com/SZUHvern/MaCo
# CheXzero: https://github.com/rajpurkarlab/CheXzero
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, LayerScale, DropPath
from timm.layers import Mlp
import torch.nn.functional as F
from torch.jit import Final
from typing import Optional

from util.pos_embed import get_2d_sincos_pos_embed
from util.anatomy_graph import init_anatomy_tree, roots2leafs
from util.misc import concat_all_gather
from bert.bert_encoder import BertEncoder
import torch.distributed as dist
from einops import rearrange
from transformers import AutoTokenizer, GPT2LMHeadModel
import tokenizers
from copy import deepcopy

def gather(embeddings, gradient=True):
    world_size = dist.get_world_size()
    embeddings = embeddings.contiguous()
    embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
    # distnn.all_gather(embeddings_list, embeddings)
    dist.all_gather(embeddings_list, embeddings)
    if gradient == True:
        embeddings_list[dist.get_rank()] = embeddings
    embeddings = torch.cat(embeddings_list)
    return embeddings

def create_logits(x1, x2, logit_scale=1):
    # logit_scale=1
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

def generate_vit_attention_mask(seq_len, k, device='cuda'):
    causal_mask = torch.full((k, k), float('-inf'), device=device)
    causal_mask.fill_diagonal_(0)
    # no_mask = torch.zeros(seq_len - k, seq_len, device=device)
    no_mask = torch.cat([torch.full((seq_len - k, k), float('-inf'), device=device), torch.zeros(seq_len - k, seq_len-k, device=device)], dim=1)
    
    mask = torch.cat([
        torch.cat([causal_mask, torch.zeros(k, seq_len - k, device=device)], dim=1),
        no_mask
    ], dim=0)

    return mask

class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather_gradient(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attention_mask=None, return_attention=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attention_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attention_mask is not None:
                # Apply the attention mask
                attn = attn + attention_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return attn
        return x
    
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attention_mask=None, return_attention=False) -> torch.Tensor:
        if return_attention:
            return self.attn(self.norm1(x), attention_mask=attention_mask, return_attention=return_attention)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attention_mask=attention_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class SSACL(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True, lam=0.9, T=0.07, SR=1.0, warmE=0, max_txt_len=100):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if SR==1.0:
            self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size * 2)**2 * in_chans, bias=True)
        elif SR == 0.0:
            self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size)**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        # Text encoder
        self.bert_encoder = BertEncoder()
        self.img_mlp = nn.Linear(embed_dim, 384, bias=True)
        self.pos_weight_img = nn.Linear(num_patches, 1)

        self.norm_pix_loss = norm_pix_loss
        self.T = 1 / T
        self.Tpar = nn.Parameter(torch.ones([]) * self.T)
        self.lam = lam
        self.SR = SR
        self.warm = warmE

        self.initialize_weights()
        self.init_pos_weight()

        # --------------------------------------------------------------------------
        # MRG components
        self.max_txt_len = max_txt_len
        self.ana_root, _, self.anatree_ids, self.anatree_nodes, self.leafnodes, self.leafids = init_anatomy_tree()
        self.anatree_ids_dict = {r:ls for (r, ls) in self.anatree_ids}

        tokenizer = tokenizers.Tokenizer.from_file("data/mimic_wordpiece.json")
        tokenizer.enable_truncation(max_length=self.max_txt_len)
        tokenizer.enable_padding(length=self.max_txt_len)

        leaf_tokens = tokenizer.encode_batch(self.leafnodes)
        self.leaf_input_ids = torch.tensor([_.ids for _ in leaf_tokens], dtype=torch.long)
        self.leaf_attention_mask = torch.tensor([_.attention_mask for _ in leaf_tokens], dtype=torch.long)
        self.leaf_type_ids = torch.tensor([_.type_ids for _ in leaf_tokens], dtype=torch.long)

        # generation model
        self.opt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.opt_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token
        self.opt_lm_vocab_size = self.opt_model.lm_head.weight.shape[0]
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            embed_dim, self.opt_model.config.hidden_size
        )
        self.bert_proj = nn.Linear(
            384, embed_dim
        )
        
    def init_pos_weight(self):
        init_weight1 = torch.linspace(0.5,1,7)
        init_weight2 = torch.linspace(1,0.5,7)
        init_weight = torch.cat([init_weight1, init_weight2], dim=-1).unsqueeze(-1)
        init_weight = init_weight @ init_weight.t()
        init_weight = init_weight.view(1, -1).softmax(dim=-1)
        
        init_weight = init_weight * 0 ## only for softplus
        self.pos_weight_img.weight = nn.Parameter(init_weight)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
       

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if self.SR == 1.0:
            p = self.patch_embed.patch_size[0]*2
        elif self.SR == 0:
            p = self.patch_embed.patch_size[0]
        
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        
        x = rearrange(x, 'n c h p w q -> n (h w) (p q c)')
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_img_encoder(self, x, mask_ratio, extra_cls_tokens=None):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if extra_cls_tokens is not None:
            extra_cls_tokens = extra_cls_tokens.expand(x.shape[0], -1, -1)
            cls_tokens = torch.cat((extra_cls_tokens, cls_tokens), 1)
            attention_mask = generate_vit_attention_mask(x.shape[1]+cls_tokens.shape[1], cls_tokens.shape[1])
        else:
            attention_mask = None
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_img_encoder_nomask(self, x, extra_cls_tokens=None):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if extra_cls_tokens is not None:
            extra_cls_tokens = extra_cls_tokens.expand(x.shape[0], -1, -1)
            cls_tokens = torch.cat((extra_cls_tokens, cls_tokens), 1)
            attention_mask = generate_vit_attention_mask(x.shape[1]+cls_tokens.shape[1], cls_tokens.shape[1])
        else:
            attention_mask = None
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attention_mask=attention_mask)
        x = self.norm(x)

        return x
    
    def forward_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_img_decoder_only(self, x, ids_restore):
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def root_query_restore(self, leaf_query):
        ## leaf to root restore
        def _restore_roots(x, root_id, leaf_ids, restored_nodes):
            leaf_feats = []
            for li in leaf_ids:
                if li in restored_nodes:
                    lf = x[:, restored_nodes.index(li)].unsqueeze(1) # [B, 1, D]
                else:
                    lf, restored_nodes, x = _restore_roots(x, li, self.anatree_ids_dict[li], restored_nodes)
                    restored_nodes.append(li)
                leaf_feats.append(lf)
            root_feats = torch.cat(leaf_feats, dim=1).mean(dim=1, keepdim=True) # [B, 1, D]
            if root_id == 0:
                restored_nodes.append(root_id)
            x = torch.cat((x, root_feats), 1)
            return root_feats, restored_nodes, x
        _, restored_ids, leaf_query = _restore_roots(leaf_query, self.anatree_ids[0][0], self.anatree_ids[0][1], deepcopy(self.leafids))
        assert len(restored_ids) == len(self.anatree_nodes), f"{leaf_query.shape} | {len(restored_ids)} | {len(self.anatree_nodes)}\n{restored_ids} | {self.anatree_nodes}"
        gathered_ids = torch.tensor([restored_ids.index(i) for i in range(len(self.anatree_nodes))], 
                                    dtype=int, device=leaf_query.device).unsqueeze(0).unsqueeze(-1).expand(leaf_query.shape[0], -1, leaf_query.shape[-1])
        leaf_query = leaf_query.gather(dim=1, index=gathered_ids)
        return leaf_query
    
    def anatomy_reconstruction(self, image_embeds, masked_image_embeds):
        loss_ar = []
        for root_id, leaf_ids in self.anatree_ids:
            leaf_embeds = masked_image_embeds[:, leaf_ids] # [B, k, D]
            root_embeds = image_embeds[:, root_id].unsqueeze(1) # [B, 1, D]
            leaf_embeds = torch.cat([leaf_embeds.mean(1, keepdim=True), leaf_embeds], 1) # [B, k+1, D]
            leaf_embeds = self.decoder_embed(leaf_embeds)
            root_embeds = self.decoder_embed(root_embeds)
            for blk in self.decoder_blocks:
                leaf_embeds = blk(leaf_embeds)
                root_embeds = blk(root_embeds)
            leaf_embeds = self.decoder_norm(leaf_embeds)[:, 0] # [B, D]
            root_embeds = self.decoder_norm(root_embeds)[:, 0] # [B, D]
            leaf_embeds = self.decoder_pred(leaf_embeds) # [B, p*p*3]
            root_embeds = self.decoder_pred(root_embeds) # [B, p*p*3]
            _loss = (leaf_embeds - root_embeds.detach()) ** 2
            _loss = _loss.mean()  # [N, L], mean loss per patch
            loss_ar.append(_loss)
        loss_ar = sum(loss_ar) / len(loss_ar)
        
        return loss_ar
    
    def anatomical_consistency_align(self, image_anat_embeds, text_embeds):
        image_anat_embeds = F.normalize(image_anat_embeds, dim=-1) # [B, k, Dv]
        image_anat_embeds_all = concat_all_gather(image_anat_embeds) # [B*n, k, Dv] detached
        text_embeds = F.normalize(text_embeds, dim=-1) # [B, Dt]
        text_embeds_all = concat_all_gather(text_embeds) # [B*n, Dt] detached

        sim_a2a = torch.matmul(
            image_anat_embeds.unsqueeze(1), image_anat_embeds_all.unsqueeze(0).permute(0, 1, 3, 2)
        ) # [B, Bn, k, k]
        sim_a2a = sim_a2a * self.Tpar
        sim_t2t = torch.matmul(
            text_embeds, text_embeds_all.permute(1, 0)
        ) # [B, Bn]
        sim_t2t = F.softmax(sim_t2t * self.Tpar, dim=-1)

        diag_matrix = torch.eye(sim_a2a.size(-1)).unsqueeze(0).unsqueeze(0).to(image_anat_embeds.device) # [1, 1, k, k]
        target_a2a = 0.9*diag_matrix + 0.1*sim_t2t.detach().unsqueeze(-1).unsqueeze(-1) # [B, Bn, k, k]

        loss_aca = -torch.sum(F.log_softmax(sim_a2a, dim=-1)*target_a2a, dim=-1).mean()

        return loss_aca
    
    def anatomy_loss(self, imgs, mask_ratio, graph_embeds, latent_leafs, global_text_embeds):
        # Anatomical Reconstruction
        latent_graph = self.forward_img_encoder_nomask(imgs, extra_cls_tokens=graph_embeds)[:, :graph_embeds.shape[1]]
        latent_graph_masked, _, _ = self.forward_img_encoder(imgs, mask_ratio, extra_cls_tokens=graph_embeds)
        latent_graph_masked = latent_graph_masked[:, :graph_embeds.shape[1]]
        loss_ar = self.anatomy_reconstruction(latent_graph, latent_graph_masked)
        
        # Anatomical Consistency Alignment
        loss_aca = self.anatomical_consistency_align(latent_leafs, global_text_embeds)

        return loss_ar, loss_aca
    
    def generation_loss(self, image_embeds_for_gen, samples, leaf_num):
        ## Autoregressive generation
        inputs_opt = self.opt_proj(image_embeds_for_gen) # [B, num_query_tokens, D]
        atts_opt = torch.zeros(inputs_opt.size()[:-1], dtype=torch.long).to(image_embeds_for_gen.device)
        atts_opt[:, leaf_num+1:] = 1 # local tokens
        anat_ids = [torch.tensor([self.leafnodes.index(aa) for aa in roots2leafs(a)], dtype=torch.long).to(image_embeds_for_gen.device) for a in samples["anatomies"]]
        for i, aids in enumerate(anat_ids):
            assert (aids < leaf_num).all(), aids
            atts_opt[i, aids] = 1 # pred anatomies
        
        self.opt_tokenizer.padding_side = "right"

        if torch.rand(1) > 0.5:
            prompt = [p + '\n' for p in samples["prompt"]]
        else:
            prompt = None
        text = [t + "\n" for t in samples["text_input"]]

        if prompt is not None:
            prompt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image_embeds_for_gen.device)
            atts_opt = torch.cat([atts_opt, prompt_tokens.attention_mask.long()], dim=1)
            
            # new version for transformers>=4.27
            inputs_prompt = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids.long())
            inputs_opt = torch.cat([inputs_opt,inputs_prompt], dim=1)

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image_embeds_for_gen.device)
        opt_tokens.input_ids, opt_tokens.attention_mask = opt_tokens.input_ids.long(), opt_tokens.attention_mask.long()

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image_embeds_for_gen.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss_lm = outputs.loss

        return loss_lm

    def forward(self, batch, mask_ratio=0.50, epoch=0):

        big_imgs, ids, labels, attention_mask, type_ids = batch["image"], batch["ids"], batch["labels"], batch["attention_mask"], batch["type_ids"]
        big_imgs = big_imgs.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()
        
        if self.SR == 1.0:
            imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(big_imgs)
        elif self.SR == 0:
            imgs = big_imgs

        latent, mask, ids_restore = self.forward_img_encoder(imgs, mask_ratio)

        ones_sample = torch.ones_like(latent, requires_grad=False)
        mask_token_fixed = torch.zeros_like(self.mask_token, requires_grad=False)
        mask_token_fixed = mask_token_fixed.repeat(ones_sample.shape[0], ids_restore.shape[1] + 1 - ones_sample.shape[1], 1)
        x_pos = torch.cat([ones_sample[:, 1:, :], mask_token_fixed], dim=1)  # no cls token
        x_pos = torch.gather(x_pos, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ones_sample.shape[2])).mean(dim=-1) # unshuffle        

        pred = self.forward_img_decoder(latent, ids_restore)  # [N, L, p*p*3]
        target = self.patchify(big_imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss_img = (pred - target) ** 2
        loss_img = loss_img.mean(dim=-1)  # [N, L], mean loss per patch
        loss_img = (loss_img * mask).sum() / mask.sum()  # mean loss on removed patches 
       
        latent_img = self.img_mlp(latent)
        latent_img = latent_img[:, 1:, :]
        latent_img_global = latent_img.mean(dim=1) 
        labels = None
        latent_report = self.bert_encoder(latent_img_global, ids, labels, attention_mask, type_ids)
        logits_report = latent_report.logits
        x_ = self.pos_weight_img(x_pos).type(torch.float32)

        latent_img_global = gather_gradient(latent_img_global)
        logits_report_gather = gather_gradient(logits_report)
        x_ = gather_gradient(x_)

        logits1, logits2 = create_logits(latent_img_global, logits_report_gather, self.Tpar)
        gt = torch.arange(logits1.shape[0], dtype=torch.long).cuda()

        self.CE = torch.nn.CrossEntropyLoss()
        scale = torch.log(1 + torch.exp(x_))
        logits1_scale = logits1 * scale
        logits2_scale = logits2 * scale
        loss_c = self.CE(logits1_scale, gt)
        loss_c += self.CE(logits2_scale, gt)
        self.WCE = torch.nn.CrossEntropyLoss(weight=scale.detach().squeeze(-1))
        loss_c += self.WCE(logits1, gt)
        loss_c += self.WCE(logits2, gt)
        loss_c /= 4

        # Anatomical Tree Embedding
        self.leaf_input_ids = self.leaf_input_ids.to(imgs.device)
        self.leaf_attention_mask = self.leaf_attention_mask.to(imgs.device)
        self.leaf_type_ids = self.leaf_type_ids.to(imgs.device)
        leaf_embeds = self.bert_encoder.model(
            input_ids=self.leaf_input_ids,
            attention_mask=self.leaf_attention_mask,
            token_type_ids=self.leaf_type_ids,
            return_dict=True,
            output_hidden_states=True,
        ).logits.unsqueeze(0) # [1, K, Dt]
        leaf_embeds = self.bert_proj(leaf_embeds)
        # root query restore
        graph_embeds = self.root_query_restore(leaf_embeds)

        # MRG
        latent_leafs = self.forward_img_encoder_nomask(imgs, extra_cls_tokens=leaf_embeds)
        loss_lm = self.generation_loss(latent_leafs, batch, leaf_num=leaf_embeds.shape[1])
        loss_ar, loss_aca = self.anatomy_loss(imgs, mask_ratio, graph_embeds=graph_embeds, 
                                              latent_leafs=latent_leafs[:, :leaf_embeds.shape[1]], global_text_embeds=logits_report)

        return (loss_img, loss_c, loss_lm, loss_ar, loss_aca), pred, mask
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=100,
        min_length=80,
        top_p=0.9,
        repetition_penalty=2.0,
        length_penalty=2.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image = image.cuda()
        # self.leaf_tokens.to(image.device)

        anatomies = samples['anatomies'] if 'anatomies' in samples else None
        if anatomies is not None:
            anat_ids = [torch.tensor([self.leafnodes.index(aa) for aa in roots2leafs(a)], dtype=torch.long).to(image.device) for a in anatomies]

        # Anatomical Tree Embedding
        self.leaf_input_ids = self.leaf_input_ids.to(image.device)
        self.leaf_attention_mask = self.leaf_attention_mask.to(image.device)
        self.leaf_type_ids = self.leaf_type_ids.to(image.device)
        leaf_embeds = self.bert_encoder.model(
            input_ids=self.leaf_input_ids,
            attention_mask=self.leaf_attention_mask,
            token_type_ids=self.leaf_type_ids,
            return_dict=True,
            output_hidden_states=True,
        ).logits.unsqueeze(0) # [1, K, Dt]
        leaf_embeds = self.bert_proj(leaf_embeds)
        
        image_embeds = self.forward_img_encoder_nomask(image, extra_cls_tokens=leaf_embeds)
    
        inputs_opt = self.opt_proj(image_embeds) # [B, (2*)num_query_tokens, D]
        atts_opt = torch.zeros(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)
        atts_opt[:, leaf_embeds.shape[1]+1:] = 1 # local tokens
        for i, aids in enumerate(anat_ids):
            assert (aids < leaf_embeds.shape[1]).all(), aids
            atts_opt[i, aids] = 1 # pred anatomies

        if "prompt" in samples:
            prompt = [p + '\n' for p in samples["prompt"]]
        else:
            prompt = None

        if prompt is not None:
            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask.long()], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids.long())
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
        else:
            inputs_embeds = inputs_opt
            attention_mask = atts_opt

        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            return_dict_in_generate=True,
            pad_token_id=self.eos_token_id,
        )
        output_text = self.opt_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
                        
        output_text = [text.strip() for text in output_text]

        report_tokens = self.opt_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        ref_reports = self.opt_tokenizer.batch_decode(report_tokens.input_ids, skip_special_tokens=True)
        ref_reports = [text.strip() for text in ref_reports]
        
        image_ids = samples['image_ids']
        output_text = {iid: t for iid, t in zip(image_ids, output_text)}
        ref_reports = {iid: t for iid, t in zip(image_ids, ref_reports)}

        return output_text, ref_reports
    
    def get_last_selfattention(self, x, extra_cls_tokens=None):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if extra_cls_tokens is not None:
            extra_cls_tokens = extra_cls_tokens.expand(x.shape[0], -1, -1)
            cls_tokens = torch.cat((extra_cls_tokens, cls_tokens), 1)
            attention_mask = generate_vit_attention_mask(x.shape[1]+cls_tokens.shape[1], cls_tokens.shape[1])
        else:
            attention_mask = None
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, attention_mask=attention_mask)
            else:
                # return attention of the last block
                return blk(x, return_attention=True, attention_mask=attention_mask)
            
    def get_hmaps(self, img, threshold=None, patch_size=16, extra_cls_tokens=False):
        if extra_cls_tokens:
            # Anatomical Tree Embedding
            self.leaf_input_ids = self.leaf_input_ids.to(img.device)
            self.leaf_attention_mask = self.leaf_attention_mask.to(img.device)
            self.leaf_type_ids = self.leaf_type_ids.to(img.device)
            leaf_embeds = self.bert_encoder.model(
                input_ids=self.leaf_input_ids,
                attention_mask=self.leaf_attention_mask,
                token_type_ids=self.leaf_type_ids,
                return_dict=True,
                output_hidden_states=True,
            ).logits.unsqueeze(0) # [1, K, Dt]
            leaf_embeds = self.bert_proj(leaf_embeds)
            # root query restore
            graph_embeds = self.root_query_restore(leaf_embeds)
            extra_cls_tokens = graph_embeds
        else:
            extra_cls_tokens = None
        
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        attentions = self.get_last_selfattention(img, extra_cls_tokens=extra_cls_tokens) # [B, nh, N, N]

        nh = attentions.shape[1] # number of head
        cls_num = extra_cls_tokens.shape[1]+1 if extra_cls_tokens is not None else 1

        attentions = attentions[0, :, :cls_num-1, cls_num:].reshape(nh, -1)

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            attentions = th_attn.reshape(nh, w_featmap, h_featmap).float().mean(0).cpu().numpy()
        else:
            attentions = attentions.reshape(nh, w_featmap, h_featmap).mean(0).cpu().numpy()
            
        return attentions

def ssacl(**kwargs):
    model = SSACL(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
