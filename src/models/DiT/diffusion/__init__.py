# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )



if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("conf/tp_config.yaml")
    diffusion = create_diffusion(timestep_respacing=cfg.diffusion.timestep_respacing, 
                                noise_schedule=cfg.diffusion.noise_schedule, 
                                use_kl=cfg.diffusion.use_kl,
                                sigma_small=cfg.diffusion.sigma_small,
                                predict_xstart=cfg.diffusion.predict_xstart,
                                learn_sigma=cfg.diffusion.learn_sigma,
                                rescale_learned_sigmas=cfg.diffusion.rescale_learned_sigmas,
                                diffusion_steps=cfg.diffusion.diffusion_steps)
    diffusion2 = create_diffusion(timestep_respacing=cfg.diffusion.timestep_respacing, 
                                noise_schedule="quad", 
                                use_kl=cfg.diffusion.use_kl,
                                sigma_small=cfg.diffusion.sigma_small,
                                predict_xstart=cfg.diffusion.predict_xstart,
                                learn_sigma=cfg.diffusion.learn_sigma,
                                rescale_learned_sigmas=cfg.diffusion.rescale_learned_sigmas,
                                diffusion_steps=cfg.diffusion.diffusion_steps)
    import matplotlib.pyplot as plt
    betas = diffusion.betas
    alphas_cumprod = diffusion.alphas_cumprod
    sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod
    betas2 = diffusion2.betas
    alphas_cumprod2 = diffusion2.alphas_cumprod
    sqrt_alphas_cumprod2 = diffusion2.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod2 = diffusion2.sqrt_one_minus_alphas_cumprod
   
    plt.figure(figsize=(10, 10), dpi=200)
    plt.plot(betas, 'r-')
    plt.plot(alphas_cumprod, 'b-')
    plt.plot(sqrt_alphas_cumprod, 'm-')
    plt.plot(sqrt_one_minus_alphas_cumprod, 'o-')
    plt.plot(betas2, 'r--')
    plt.plot(alphas_cumprod2, 'b--')
    plt.plot(sqrt_alphas_cumprod2, 'm--')
    plt.plot(sqrt_one_minus_alphas_cumprod2, 'o--')
    plt.legend(["lin_beta", "lin_alpha", "lin_alpha_cumprod", "1-lin_alpha_cumprod",
                "quad_beta", "quad_alpha", "quad_alpha_cumprod", "1-quad_alpha_cumprod"])
    plt.savefig("/home/lianghongli/caiyun-algo-misc/DiT/ddpm_schedule.png")
    
