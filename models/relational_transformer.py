import math
import torch
import torch.nn as nn
import timm

from models.vision_transformer import Block, partial, _init_vit_weights, trunc_normal_, named_apply


class RelationalTransformer(nn.Module):

    def __init__(self, input_dim, num_classes=1, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_rate=0, attn_drop_rate=0, drop_path_rate=0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_classes = num_classes

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward(self, x, return_attention=False, return_feats=False):
        # x is a concat of the features of 2 images, we should separate them into 2 tokens
        x1, x2 = torch.chunk(x,2,dim=1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        # final sequence will be: [cls_token, img1_token, img2_token]
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # append class token
        x = torch.cat((cls_token, x1, x2), dim=1)
        
        x = self.pos_drop(x)
        attn = []
        for idx, blk in enumerate(self.blocks):
            x, attn_blk = blk(x)
            attn.append(attn_blk)

        x = self.norm(x)

        # take only cls_token output
        feats = x[:, 0]
        out = self.head(feats)
        output = (out,)
        if return_feats:
            output += (feats,)
        if return_attention:
            output += (attn,)
        if len(output) == 1:
            return output[0]
        return output
