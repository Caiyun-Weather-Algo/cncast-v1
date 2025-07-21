# caiyun-cncast-v1

A collection of weather forecast for limited areas.

## CNcast-v1: Leveraging 3D Swin Transformer and DiT for Enhanced Regional Weather Forecasting

CNcast-v1 is a state-of-the-art deep learning model for regional weather forecasting, designed to provide high-accuracy hourly regional weather states up to 5 days. The model is based on the Swin Transformer 3D, a self-attention mechanism that can capture long-range dependencies in the input data. The model can be used to provide real-time weather forecasts for a wide range of applications, including weather-related decision-making, disaster response, and resource allocation.

### Architecture

The model architectures for weather prediction and precipitation diagnosis are illustrated below:

![CNcast-v1 Architecture](assets/swin3d.bmp)
![Precipitation diagnosis Architecture](assets/dit-precip-flow.bmp)

### Key Features

- Hourly regional weather forecasting up to 5 days with enhanced precipitation prediction
- High spatial resolution of approximately 5 kilometers
- Enhanced boundary conditions inspired by traditional NWP techniques
- Initialized with IFS open data
- Integration of latent diffusion model for high spatial resolution of precipitation diagnostic forecasting


### Getting Started

### Installation
1. Install the necessary modules by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Training
- **Non-TP Variable Prediction**: Train the model for non-tp variable predictions using the following command:
  ```bash
  python train_tpu.py
  ```
- **VAE for CMPAS/ERA5/ERA5_tp training**: Train the VAE models for different variables using the following command(coming sson):
  ```bash
  python train_gpu_vae.py
  ```
- **DiT Model for Total Precipitation Diagnosis**: Train the DiT model to diagnose total precipitation using non-tp variables with the command below(coming soon):

  ```bash
  python train_tp.py
  ```

### Evaluation
The codes to reproduce the results demonstrated in the paper are located in the `src/eval/` directory. To make predictions based on 2021 ERA5 data or to make near real-time predictions based on IFS, run the following command:

  ```bash
  python inference.py --forecast_day 5 --data_source era5 --data_format zarr
  python inference.py --forecast_day 5 --data_source ifs --data_format grib2
  ```

### Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{cncast2025,
    title={CNCast: Leveraging 3D Swin Transformer and DiT for Enhanced Regional Weather Forecasting},
    author={Liang, Hongli and Zhang, Yuanting and Meng, Qingye and He, Shuangshuang and Yuan, Xingyuan},
    journal={arXiv preprint arXiv:2503.13546},
    year={2025},
    url={https://arxiv.org/abs/2503.13546}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact

For questions and collaborations, you can:

- Open an issue in this repository
- Contact the authors directly:
  - Hongli Liang: lianghongli@caiyunapp.com
  - Yuanting Zhang: zhangyuanting@caiyunapp.com

We welcome feedback, questions, and potential collaboration opportunities.

