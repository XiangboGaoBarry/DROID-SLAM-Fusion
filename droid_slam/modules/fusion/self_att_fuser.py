import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.fusion.transformer import myTransformerBlock

class SelfAttFuser(nn.Module):

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=64, horz_anchors=32,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), feat1 + feat2
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.avgpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):

        feat1 = x[0]  # feat1 (tensor): dim:(B, C, H, W)
        feat2 = x[1]   # feat2 (tensor): dim:(B, C, H, W)
        assert feat1.shape[0] == feat2.shape[0]
        if len(feat1.shape) == 5:
            bs, c, num_frames, h, w = feat1.shape
            feat1 = feat1.reshape(bs,c*num_frames, h,w)    
            feat2 = feat2.reshape(bs,c*num_frames, h,w)    
        bs, c, h, w = feat1.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        feat1 = self.avgpool(feat1)
        feat2 = self.avgpool(feat2)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        feat1_flat = feat1.view(bs, c, -1)  # flatten the feature
        feat2_flat = feat2.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([feat1_flat, feat2_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        feat1_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        feat2_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        feat1_out = F.interpolate(feat1_out, size=([h, w]), mode='bilinear')
        feat2_out = F.interpolate(feat2_out, size=([h, w]), mode='bilinear')

        return feat1_out, feat2_out
