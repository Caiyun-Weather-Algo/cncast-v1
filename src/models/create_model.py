from src.models import swin_transformer_3d_slidepatch, swinir
from src.models.DiT import dit
from src.models import autoencoder


def load_model(model_name, cfg):
    if model_name=="swin_transformer_3d_slidepatch":
        args = cfg.model.swin_transformer_3d_slidepatch
        model = swin_transformer_3d_slidepatch.SwinTransformer3D(in_chans=args.in_chans,
                              patch_size=args.patch_size,
                              embed_dim=args.embed_dim,
                              window_size=args.window_size,
                              depths=args.depths,
                              num_heads=args.num_heads, 
                              add_boundary=args.add_boundary,
                              use_checkpoint=args.use_checkpoint, 
                              )
    elif model_name=="dit":
        args = cfg.model.dit
        model = dit.DiT_models["DiT-S/4"](input_size=args.input_size, cond_size=args.cond_size, img_size=args.img_size, 
                                        in_channels=args.in_channels, out_channels=args.out_channels, 
                                        cond_channels=args.cond_channels, learn_sigma=args.learn_sigma)
    elif model_name=="dit_cmpa":
        args = cfg.model.dit_cmpa
        model = DiT_models["DiT-S/2"](
            input_size=input_size,
            cond_size=args.latent_shape[1:],
            img_size=args.img_size,
            in_channels=args.in_channels,
            cond_channels=args.cond_channels, 
            out_channels=args.out_channels,
            learn_sigma=args.learn_sigma,
        )
    elif model_name=="swinir":
        args = cfg.model.swinir
        model = swinir.SwinIR(upscale=args.upscale, img_size=args.img_size, in_chans=args.in_chans, out_chans=args.out_chans,
                   window_size=args.window_size, patch_size=args.patch_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='nearest+conv')
    elif model_name=="era5_tp_autoencoder":
        args = cfg.model.era5_tp_autoencoder
        model = autoencoder.AutoencoderKL(
                ddconfig=args.ddconfig, 
                lossconfig=args.lossconfig, 
                embed_dim=args.embed_dim, 
                ckpt_path=args.ckpt_path,
                )
    elif model_name=="era5_autoencoder":
        args = cfg.model.era5_autoencoder
        model = autoencoder.AutoencoderKL(
                ddconfig=args.ddconfig, 
                lossconfig=args.lossconfig, 
                embed_dim=args.embed_dim, 
                ckpt_path=args.ckpt_path,
                )
    elif model_name=="cmpa_autoencoder":
        args = cfg.model.cmpa_autoencoder
        model = autoencoder.AutoencoderKL(
                ddconfig=args.ddconfig, 
                lossconfig=args.lossconfig, 
                embed_dim=args.embed_dim, 
                ckpt_path=args.ckpt_path,
        )
    return model


def main():
    ###### test model ########
    import yaml
    from omegaconf import DictConfig
    import torch

    cfg = DictConfig(yaml.safe_load(open("./configs/train.yaml")))
    ''' test dit loading'''
    # model = load_model("dit", cfg=cfg)
    # size = [256, 256]
    # x = torch.randn(1, 69, *size)
    # y = torch.randn(1, 1, *size)
    # t = torch.randn(1)
    # output = model(y, t, x)
    # print(output.shape)
    ''' test swin3d loading'''
    # model = load_model("swin_transformer_3d_slidepatch", cfg=cfg)
    # size = [241, 281]
    # x0 = torch.rand(1,5,*size)#.cuda()
    # x1 = torch.rand(1,5*13, *size)#.cuda()
    # # x = x.cuda()
    # # print('input', x.shape)
    # x = torch.cat((x0, x1), dim=1)
    # surf_bc_bottom = torch.rand(1,4,4, 281)
    # surf_bc_top = torch.rand(1,4,4,281)
    # surf_bc_left = torch.rand(1,4,241,4)
    # surf_bc_right = torch.rand(1,4,241,4)
    # high_bc_bottom = torch.rand(1,5,13,4,281)
    # high_bc_top = torch.rand(1,5,13,4,281)
    # high_bc_left = torch.rand(1,5,13,241,4)
    # high_bc_right = torch.rand(1,5,13,241,4)
    # x_bc=[[surf_bc_bottom, surf_bc_left, surf_bc_top, surf_bc_right], [high_bc_bottom, high_bc_left, high_bc_top, high_bc_right]]
    # output = model(x, x_bc)
    # print(output[0].shape, output[1].shape)
    ### test swinir loading
    # model = load_model("swinir", cfg=cfg)
    # ckpt = torch.load(cfg.directory.model_path + "/cn_tp_0p25/swinir_tp_sr.pth.tar")["model"]
    # model.load_state_dict(ckpt)
    # x = torch.rand(1, 1, 256, 256)
    # output = model(x)
    # print(output.shape)
    ''' test vae model loading '''
    ''' 1. ERA5 VAE loading '''
    model = load_model("era5_autoencoder", cfg=cfg)
    x = torch.rand((1, 69, 181, 281))
    y = model(x)
    print("ERA5 VAE output shape:", y[0].shape)
    ''' 2. CMPA VAE loading '''
    model = load_model("cmpa_autoencoder", cfg=cfg)
    x = torch.rand((1, 1, 900, 1400))
    y = model(x)
    print("CMPA VAE output shape:", y[0].shape)
    ''' 3. ERA5 TP VAE loading '''
    model = load_model("era5_tp_autoencoder", cfg=cfg)
    x = torch.rand((1, 1, 181, 281))
    y = model(x)
    print("ERA5 TP VAE output shape:", y[0].shape)


if __name__ == '__main__':
    main()
