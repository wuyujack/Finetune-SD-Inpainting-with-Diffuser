export MODEL_DIR="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="./save/no_text_lr_1e-5_10epoch_debug_dataaug_addcloth/"

accelerate launch train_controlnet_stablediffusion_inpaint_with_cloth_channel.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --data_dir="/data1/mingfuliang/VITON-HD/" \
 --checkpointing_steps 500 \
 --num_train_epochs=1 \
 --resolution=512 \
 --learning_rate=1e-5 \
 --controlnet_model_name_or_path "shgao/edit-anything-v0-4-sd21" \
 --validation_image "/data1/mingfuliang/VITON-HD/example/conditioning_image_1.jpg" "/data1/mingfuliang/VITON-HD/example/conditioning_image_2.jpg" \
 --validation_prompt "" "" \
 --validation_mask "/data1/mingfuliang/VITON-HD/example/mask_1.jpg" "/data1/mingfuliang/VITON-HD/example/mask_2.jpg" \
 --validation_masked_image "/data1/mingfuliang/VITON-HD/image/00000_00.jpg" "/data1/mingfuliang/VITON-HD/image/00001_00.jpg" \
 --train_batch_size=1 \
 --enable_xformers_memory_efficient_attention \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --set_grads_to_none