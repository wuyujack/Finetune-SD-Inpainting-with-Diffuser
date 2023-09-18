export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="./save_SDC_scale_with_text/with_text_lr_1e-5_10epoch_ccs1/"

accelerate launch train_controlnet_stablediffusion_inpaint_with_text.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --data_dir="/data1/mingfuliang/VITON-HD/" \
 --controlnet_conditioning_scale 1.0 \
 --checkpointing_steps 500 \
 --num_train_epochs 10 \
 --resolution 512 \
 --learning_rate=1e-5 \
 --controlnet_model_name_or_path "shgao/edit-anything-v0-4-sd21" \
 --validation_image "/data1/mingfuliang/VITON-HD/example/conditioning_image_1.jpg" "/data1/mingfuliang/VITON-HD/example/conditioning_image_2.jpg" \
 --validation_prompt "the model wears a levi's logo white t-shirt" "the model wears a black t - shirt and leather pants" \
 --validation_mask "/data1/mingfuliang/VITON-HD/example/mask_1.jpg" "/data1/mingfuliang/VITON-HD/example/mask_2.jpg" \
 --validation_masked_image "/data1/mingfuliang/VITON-HD/image/00000_00.jpg" "/data1/mingfuliang/VITON-HD/image/00001_00.jpg" \
 --train_batch_size 1 \
 --validation_steps 50 \
 --enable_xformers_memory_efficient_attention \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --use_8bit_adam \
