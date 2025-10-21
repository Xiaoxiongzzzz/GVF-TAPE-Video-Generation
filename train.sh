accelerate launch --config_file ./config.yaml\
    --main_process_port 29502 \
    flowdiffusion/train_rectified_flow_real_world.py \
    --use-wandb \
    # --depth \
    # --fine-tune \
    # --model-path /mnt/data0/xiaoxiong/single_view_goal_diffusion/real_world_results/RFlow_CORL_HAND_scratch_100k_depth_128_depth/ckpt/model_100000.pt 

# accelerate launch --config_file ./config.yaml\
#     --main_process_port 29502 \
#     flowdiffusion/train_rectified_flow.py \
#     --use-wandb \
    # --depth \
    # --fine-tune \
    # --model-path /mnt/data0/xiaoxiong/single_view_goal_diffusion/results/RFlow_libero_90_depth/ckpt/model_100000.pt