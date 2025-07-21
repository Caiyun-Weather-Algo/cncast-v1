# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/mmaction/models/backbones/swin_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
 
from src.models.model_helper import trunc_normal_
from src.models.modules.patch_process import PatchEmbed3D, PatchEmbed2D, PatchMerging, Upsampling
from src.models.modules.swin3d_blocks import BasicLayer

#@BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=[3,4,1],
                 var_depth=7,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False, 
                 add_boundary=False, 
                 fcst_step=6,
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.add_boundry = add_boundary
        self.in_chans = in_chans

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans[1], embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed2d = PatchEmbed2D(
            patch_size=patch_size[1:], in_chans=in_chans[0]+in_chans[2], embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.add_boundry:
            self.bc_surfs = nn.ModuleList()
            self.bc_highs = nn.ModuleList()
            if patch_size[-1]==4:
                patches_surf = [(4,4), (4,4), (4,4), (4,4)]
                patches_high = [(2,4,4), (2,4,4), (2,4,4), (2,4,4)]
            else:
                patches_surf = [(4,2), (2,4), (4,2), (2,4)]
                patches_high = [(2,4,2), (2,2,4), (2,4,2), (2,2,4)]
            for i in range(4):
                self.bc_surfs.append(PatchEmbed2D(
                    patch_size=patches_surf[i], in_chans=in_chans[0], embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None), 
                    )
                self.bc_highs.append(PatchEmbed3D(
                    patch_size=patches_high[i], in_chans=in_chans[1], embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None), 
                    )
            self.bc_surf_weight = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            self.bc_high_weight = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        downsamples = [None, PatchMerging, None]
        upsamples = [None, Upsampling, None]
        embed_dims = [embed_dim, embed_dim*2, embed_dim]
        #downsamples = [None, None]
        #upsamples = [None, None]
        #embed_dims = [embed_dim, embed_dim]
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                #dim=int(embed_dim * 2**i_layer),
                dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsamples[i_layer],
                upsample=upsamples[i_layer],
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        self.gelu = nn.GELU()

        #unembed
        self.unembed = nn.Linear(embed_dim, patch_size[0]*patch_size[1]*patch_size[2]*in_chans[1]*1, bias=False)
        self.unembed2d = nn.Linear(embed_dim, patch_size[1]*patch_size[2]*in_chans[0]*1, bias=False)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        #if isinstance(self.pretrained, str):
        #    self.apply(_init_weights)
        #    logger = get_root_logger()
        #    logger.info(f'load model from: {self.pretrained}')

        #    if self.pretrained2d:
        #        # Inflate 2D model into 3D model.
        #        self.inflate_weights(logger)
        #    else:
        #        # Directly load 3D model.
        #        #load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        #elif self.pretrained is None:
        #    self.apply(_init_weights)
        #else:
        #    raise TypeError('pretrained must be a str or None')

    def forward(self, X, x_bc=None):
        """Forward function."""
        in_chans2d = self.in_chans[0]+self.in_chans[2]
        x0 = X[:,:in_chans2d]
        x1 = rearrange(X[:,in_chans2d:], 'b (v l) h w -> b v l h w', v=self.in_chans[1])
        B,C,D,H,W = x1.shape
        x0 = self.patch_embed2d(x0)  # shape: n,embed_dim, 64, 64
        x1 = self.patch_embed(x1) # shape: n,embed_dim, 4, 64, 64
        if x_bc is not None:
            surf_bc, high_bc = x_bc
            _,_,h,w = x0.shape
            surf_bc_bottom = self.bc_surfs[0](surf_bc[0])  # shape: n, embed_dim, 1, 256
            surf_bc_left = self.bc_surfs[1](surf_bc[1])  # shape: n, embed_dim, 1, 256
            surf_bc_top = self.bc_surfs[2](surf_bc[2])  # shape: n, embed_dim, 1, 256
            surf_bc_right = self.bc_surfs[3](surf_bc[3])  # shape: n, embed_dim, 1, 256
            high_bc_bottom = self.bc_highs[0](high_bc[0])  # shape: n, embed_dim, 4, 1, 256
            high_bc_left = self.bc_highs[1](high_bc[1])  # shape: n, embed_dim, 4, 1, 256
            high_bc_top = self.bc_highs[2](high_bc[2])  # shape: n, embed_dim, 4, 1, 256
            high_bc_right = self.bc_highs[3](high_bc[3])  # shape: n, embed_dim, 4, 1, 256
            #######################
            x0 = torch.cat((surf_bc_top, x0, surf_bc_bottom), dim=-2)
            x0 = torch.cat((F.pad(surf_bc_left, (0,0,1,1), mode="replicate"), x0, F.pad(surf_bc_right, (0,0,1,1), mode="replicate")), dim=-1)
            x1 = torch.cat((high_bc_top, x1, high_bc_bottom), dim=-2)
            x1 = torch.cat((F.pad(high_bc_left, (0,0,1,1,0,0), mode="replicate"), x1, F.pad(high_bc_right, (0,0,1,1,0,0), mode="replicate")), dim=-1)
            x = torch.cat((x0.unsqueeze(2), x1), dim=2)
                    
        else:
            x = torch.cat((x0.unsqueeze(2), x1), dim=2)
        #print('patch_embed', x.shape)
        # [8, 96, 7, 64, 64]
        x = self.pos_drop(x)

        x0 = self.layers[0](x.contiguous())
        x1 = self.layers[1](x0.contiguous())
        x = x0+x1[...,:-1,:-1]
        x = self.layers[2](x.contiguous())
        x = x0+x  ## double residual
        if x_bc is not None:
            x = x[...,1:-1,1:-1]
        # unembed
        # x.shape [8, 96, 7, 64, 64]
        x = rearrange(x, 'n c d h w -> n d h w c') # 5
        x0 = x[:,0]
        x0 = self.unembed2d(x0)
        x0 = rearrange(x0, 'n h w (h0 w0 c) -> n (h h0) (w w0) c', h0=self.patch_size[1], w0=self.patch_size[2])
        x0 = rearrange(x0, 'n h w c -> n c h w')
        x1 = self.unembed(x[:,1:])
        x1 = rearrange(x1, 'n d h w (d0 h0 w0 c) -> n (d d0) (h h0) (w w0) c', d0=self.patch_size[0], h0=self.patch_size[1], w0=self.patch_size[2])
        if self.patch_size[-1]==4:
            x0 = x0[:,:,1:-2,1:-2]
        else:
            x0 = x0[:,:,:-1,:-1]
        x1 = x1[:,:-1]
        x1 = rearrange(x1, 'n d h w c -> n (c d) h w')
        if self.patch_size[-1]==4:
            x1 = x1[...,1:-2,1:-2]
        else:
            x1 = x1[...,:-1,:-1]
        out = torch.cat((x0,x1), dim=1)
        return out

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()
        

if __name__ == '__main__':
    model = SwinTransformer3D(in_chans=[4, 5, 1], 
                              patch_size=(2,2,2), 
                              embed_dim=96, 
                              var_depth=7,
                              window_size=(2,7,7), 
                              depths=[3, 9, 3], 
                              fcst_step=6, 
                              num_heads=[3, 6, 12, 24], 
                              add_boundary=True
                              )
    # model.cuda()
    # x = torch.rand((16,5*8,256,256)) # B, C, D, H, W
    x0 = torch.rand(1,5,241, 281)#.cuda()
    x1 = torch.rand(1,5*13, 241, 281)#.cuda()
    # x = x.cuda()
    # print('input', x.shape)
    x = torch.cat((x0, x1), dim=1)
    surf_bc_bottom = torch.rand(1,4,4, 281)
    surf_bc_top = torch.rand(1,4,4,281)
    surf_bc_left = torch.rand(1,4,241,4)
    surf_bc_right = torch.rand(1,4,241,4)
    high_bc_bottom = torch.rand(1,5,13,4,281)
    high_bc_top = torch.rand(1,5,13,4,281)
    high_bc_left = torch.rand(1,5,13,241,4)
    high_bc_right = torch.rand(1,5,13,241,4)
    y = model(x, x_bc=[[surf_bc_bottom, surf_bc_left, surf_bc_top, surf_bc_right], [high_bc_bottom, high_bc_left, high_bc_top, high_bc_right]])
    print('out', y.shape)
