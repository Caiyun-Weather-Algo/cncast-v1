import os
import torch
import torch_xla.core.xla_model as xm


def save_model(epoch, model, optimizer, scheduler, train_loss, valid_loss, model_path, cfg, best=False):
    if best:
        ckpt_file = f'{model_path}/{cfg.model.name}_consolidated_best.pth.tar'
    else:
        ckpt_file = f'{model_path}/{cfg.model.name}_epoch{epoch}_consolidated.pth.tar'
    model.eval()
    ckpt = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),  
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": valid_loss,
        "train_loss": train_loss, 
        "epoch": epoch,
    }
    torch.save(ckpt, ckpt_file)  
    print(f'checkpoint saved to {ckpt_file}')
    return


def save_gan_model(epoch, models, optimizers, schedulers, train_loss, valid_loss, model_path, cfg, best=False):
    optimizer_G, optimizer_D = optimizers
    scheduler_G, scheduler_D = schedulers
    if best:
        ckpt_file = f'{model_path}/{cfg.model.name}_consolidated_best.pth.tar'
    else:
        ckpt_file = f'{model_path}/{cfg.model.name}_epoch{epoch}_consolidated.pth.tar'
    models[0].eval()
    models[1].eval()
    ckpt = {
        'model_G_state_dict': models[0].module.state_dict(),
        'model_D_state_dict': models[1].module.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),  
        "scheduler_G_state_dict": scheduler_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),  
        "scheduler_D_state_dict": scheduler_D.state_dict(),
        "val_loss": valid_loss,
        "train_loss": train_loss, 
        "epoch": epoch,
    }
    torch.save(ckpt, ckpt_file)  
    print(f'checkpoint saved to {ckpt_file}')
    return


def load_training_ckpt(cfg, ckpt_path, ckpt_suffix="_rank-*-of-*.pth"):
    fs = os.listdir(ckpt_path)
    if len(fs)<1:
        return [None, 1]
    epochs = sorted([eval(f.split(f"{cfg.model.name}_")[1].split("_")[0].split("epoch")[1]) for f in fs if "epoch" in f])  ##TODO revise path
    start_epoch = epochs[-1]
    ckpt_file = f"{ckpt_path}/{cfg.model.name}_epoch{start_epoch}_consolidated.pth.tar"
    print(ckpt_file)
    start_epoch += 1
    ## model name example: swin_transformer_3d_epoch18_rank-00000007-of-00000008.pth
    ckpt = torch.load(ckpt_file, map_location="cpu")
    print("ckpt loaded", ckpt.keys())
    return [ckpt, start_epoch]


##TODO: revise or remove this function
def get_model(cfg):
    if cfg.model.era5_vae is not None:
        args = cfg.model.tp_autoencoder
        # era5_vae = autoencoder_kl.AutoencoderKL(in_channels = args.in_channels,
        #             out_channels = args.out_channels,
        #             down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D",),
        #             up_block_types = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D",),
        #             block_out_channels = args.block_out_channels,
        #             layers_per_block = 1,
        #             act_fn = "silu",
        #             latent_channels = args.latent_channels,
        #             norm_num_groups = 32,
        #             scaling_factor = args.scaling_factor,
        #             sample_size = args.sample_size,
        #             )
        tp_vae = autoencoder.AutoencoderKL(
                ddconfig=cfg.model.tp_autoencoder.ddconfig, 
                lossconfig=cfg.model.tp_autoencoder.lossconfig, 
                embed_dim=cfg.model.tp_autoencoder.embed_dim, 
                )
        era5_vae = autoencoder.AutoencoderKL(
                ddconfig=cfg.model.era5_autoencoder.ddconfig, 
                lossconfig=cfg.model.era5_autoencoder.lossconfig, 
                embed_dim=cfg.model.era5_autoencoder.embed_dim, 
                )
    else:
        era5_vae = None
    if cfg.model.cmpa_vae is not None:
        cmpa_vae = autoencoder.AutoencoderKL(
                ddconfig=cfg.model.cmpa_autoencoder.ddconfig, 
                lossconfig=cfg.model.cmpa_autoencoder.lossconfig, 
                embed_dim=cfg.model.cmpa_autoencoder.embed_dim, 
                )
    else:
        cmpa_vae = None
    if "DiT" in cfg.model.name:
        if cfg.dataload.cut_era5:
            if cfg.dataload.target=="cmpa":
                in_channels = cfg.model.cmpa_autoencoder.latent_shape[0]
                img_size = input_size = cfg.model.cmpa_autoencoder.latent_shape[1:]
                out_channels = cfg.model.cmpa_autoencoder.latent_shape[0]
                cond_channels = cfg.model.era5_autoencoder.latent_shape[0]+cfg.model.cmpa_autoencoder.latent_shape[0]*(cfg.dataload.cmpa_frame-1)+cfg.model.tp_autoencoder.latent_shape[0]*cfg.dataload.tp_era5_seperate_encode
                print("cond_channels: ", cond_channels)
            else:
                in_channels = cfg.model.era5_autoencoder.latent_shape[0]
                img_size = input_size = cfg.model.era5_autoencoder.latent_shape[1:]
                out_channels = cfg.model.era5_autoencoder.latent_shape[0]
                cond_channels = cfg.model.era5_autoencoder.latent_shape[0]+cfg.model.tp_autoencoder.latent_shape[0]*cfg.dataload.tp_era5_seperate_encode
        else:
            in_channels = cfg.model.era5_autoencoder.ddconfig.in_channels
            img_size = input_size = (240, 280)
            out_channels = cfg.model.era5_autoencoder.ddconfig.out_ch
            cond_channels = cfg.model.era5_autoencoder.ddconfig.in_channels

        model = DiT_models[cfg.model.name](
            input_size=input_size,
            cond_size=cfg.model.era5_autoencoder.latent_shape[1:],
            img_size=img_size,
            in_channels=in_channels,
            cond_channels=cond_channels, 
            out_channels=out_channels,
            learn_sigma=cfg.diffusion.learn_sigma,
        )
    return model, era5_vae, cmpa_vae, tp_vae 
