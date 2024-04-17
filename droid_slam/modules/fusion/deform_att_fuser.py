import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.fusion.deformable_attention import deformable_attn_pytorch, \
    LearnedPositionalEncoding, constant_init, xavier_init
import math
import warnings


class DeformableSpatialAttentionLayer(nn.Module):
    def __init__(self, 
                 embed_dims,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1):
        super(DeformableSpatialAttentionLayer, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # however, CUDA is not available in this implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.'
                'However, CUDA is not available in this implementation.')
            
        assert dim_per_head % 2 == 0, "embed_dims must be divisible by 2"
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(self.embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.init_weights()
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, 1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # TODO: Remove the hard coded half precision
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None,):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_query, embed_dims).
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            spatial_shapes (tuple): Spatial shape of features (h, w).

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        
        bs, num_query, _ = query.shape
        h, w = spatial_shapes
        
        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(bs, num_query, self.num_heads, -1) # bs, num_query, num_heads, embed_dims//num_heads
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_points, 2) # bs, num_query, num_heads, num_points, 2
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_points) # bs, num_query, num_heads, num_points
        attention_weights = attention_weights.softmax(-1).to(dtype) # TODO: attention_weights.softmax(-1) changed attention_weights from half to float
        
        reference_points = self.get_reference_points(h, w, bs=bs, device=device, dtype=dtype) # bs, num_query, 2
        offset_normalizer = torch.Tensor([w, h]).to(device).to(dtype)
        sampling_locations = reference_points[:, :, None, None, :] \
            + sampling_offsets / offset_normalizer
        
        output = self.output_proj(deformable_attn_pytorch(value, (h, w), sampling_locations, attention_weights))
        
        # return self.dropout(output) + identity
        return self.dropout(output) + identity
        
    
    def get_reference_points(self, H, W, bs=1, device='cuda', dtype=torch.half):
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1)
        return ref_2d

class DeformableSpatialAttentionModule(nn.Module):
    def __init__(self, 
                 embed_dims,
                 H=256,
                 W=256,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1,
                 num_frames=3):
        super(DeformableSpatialAttentionModule, self).__init__()
        self.embed_dims = embed_dims
        self.positional_encoding = LearnedPositionalEncoding(embed_dims//2, H, W)
        self.attention_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.attention_layers.append(DeformableSpatialAttentionLayer(embed_dims, num_heads, num_points, dropout))
    
    def forward(self,
                layer,
                query,
                key=None,
                value=None,
                identity=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None):
        
        bs, num_query, embed_dims = query.shape
        h, w = spatial_shapes
        pos_mask = torch.zeros((bs, h, w), device=device).to(dtype)
        query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # bs, num_query, embed_dims=pos_dim*2
        
        return self.attention_layers[layer](query=query,
                                            key=key,
                                            value=value,
                                            query_pos=query_pos,
                                            identity=identity,
                                            device=device,
                                            dtype=dtype,
                                            spatial_shapes=spatial_shapes)
        


class DeformAttFuser(nn.Module):
    """Deformable Attention Module."""
    
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=8,
                 num_points=12,
                 dropout=0.1):
        super(DeformAttFuser, self).__init__()
        self.n_layers = n_layers
        self.embed_dims = embed_dims
        self.H = H
        self.W = W
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = dropout
        self.initModule()
        
    def initModule(self):
        self.attention1 = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout)
        self.attention2 = DeformableSpatialAttentionModule(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.dropout)
    
    def forward(self, x):
        '''
        Args:
            x (tuple)
        return:
        '''
        feat1 = x[0]  # feat1 (tensor): dim:(B, C, H, W)
        feat2 = x[1]   # feat2 (tensor): dim:(B, C, H, W)
        assert feat1.shape[0] == feat2.shape[0]
        bs, c, h, w = feat1.shape
        
        feat1 = feat1.reshape(bs, c, h*w).transpose(1, 2)  # bs, h*w, embed_dims
        feat2 = feat2.reshape(bs, c, h*w).transpose(1, 2)  # bs, h*w, embed_dims
    
        for layer in range(self.n_layers):
            feat1_out = self.attention1(layer=layer,
                                            query=feat1, 
                                            key=feat2, 
                                            value=feat2, 
                                            identity=feat1, 
                                            device=feat1.device, 
                                            dtype=feat1.dtype,
                                            spatial_shapes=(h, w))
                
            feat2_out = self.attention2(layer=layer,
                                            query=feat2, 
                                            key=feat1, 
                                            value=feat1, 
                                            identity=feat2,
                                            device=feat2.device,
                                            dtype=feat2.dtype,
                                            spatial_shapes=(h, w))
                

            
            feat1 = feat1_out
            feat2 = feat2_out
                
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        feat1 = feat1.transpose(1,2).reshape(bs, c, h,w)
        feat2 = feat2.transpose(1,2).reshape(bs, c, h,w)
        
        return feat1, feat2
    
        