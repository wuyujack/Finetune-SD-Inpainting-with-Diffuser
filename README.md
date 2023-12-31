Some code I implemented for the course project of [CS496 Deep Generative Models](https://interactiveaudiolab.github.io/teaching/generative_deep_models.html). The main increment compared to [diffuser](https://github.com/huggingface/diffusers) is to support finetuning on the `controlnet + stable diffusion` model for virtual try-on tasks, which including extending the input dimension of the stable diffusion model and fully tune the whole stable diffusion model with controlnet.

The code has not been fully organized, and will not be actively maintained in the future. I release here just want to provide an example to demonstrate how we can adapt the existing Dreambooth inpainting code in diffuser to do finetuning on `controlnet + stable diffusion`, and how we can develop such a training pipeline with minimal efforts. 

As an early exploration, the finetuning results are not good for virtual try-on, where the reasons have demonstrated in our blog post [here](https://ukaukaaaa.github.io/viton.html) on Image-guided VITON with diffusion model. Given that this is only a course final project developped in three day, we only focus on promptly validating our idea instead of pursuing for the state-of-the-art results showing in exisitng VITON literature, thus we do not put so much effort on dataset selection, image preprocessing, hyperparameter tunning, and changing the whole methodology and network architecture.

To use the code, please refer to the folder of `/example/controlnet/`. The training command are the .sh files with `run_` in front of the file name, e.g., `run_stable_diffusion_controlnet_inpaint.sh`.

I use the VITON-HD dataset by defaulted and has done some postprocessing for training, where you can download the post-processed dataset from [here](https://drive.google.com/file/d/1SEck0NoSIttSpCu0wfYgHIEl5i-tZWjH/view?usp=sharing).

For the configuration of the environment, please refer to the `environment.yml`.