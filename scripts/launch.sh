####################################################################
# Training EDM models on class-conditional ImageNet-64, and LSUN 256
####################################################################

mpiexec -n 8 python edm_train.py --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 4096 --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --weight_decay 0.0 --weight_schedule karras --data_dir /path/to/imagenet

python -m orc.diffusion.scripts.train_imagenet_edm --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.0001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --use_scale_shift_norm False --weight_decay 0.0 --weight_schedule karras --data_dir /path/to/lsun_bedroom

#########################################################################
# Sampling from EDM models on class-conditional ImageNet-64, and LSUN 256
#########################################################################

mpiexec -n 8 python image_sample.py --training_mode edm --batch_size 64 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path edm_imagenet64_ema.pt --attention_resolutions 32,16,8  --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras

mpiexec -n 8 python image_sample.py --training_mode edm --generator determ-indiv --batch_size 8 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras

#########################################################################
# Consistency distillation on class-conditional ImageNet-64, and LSUN 256
#########################################################################

## L_CD^N (l2) on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data

## L_CD^N (LPIPS) on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data

## L_CD^N (l2) on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256

## L_CD^N (LPIPS) on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256

#########################################################################
# Consistency training on class-conditional ImageNet-64, and LSUN 256
#########################################################################

## L_CT^N on ImageNet-64
mpiexec -n 8 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 800000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/imagenet64

## L_CT^N on LSUN 256
mpiexec -n 8 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 150 --total_training_steps 1000000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 --lr 0.00005 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/bedroom256

#################################################################################
# Sampling from consistency models on class-conditional ImageNet-64, and LSUN 256
#################################################################################

## ImageNet-64
mpiexec -n 8 python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler onestep --model_path /path/to/checkpoint --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## LSUN-256
mpiexec -n 8 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path /root/consistency/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 100 --resblock_updown True --use_fp16 True --weight_schedule uniform

######################################################################################
# Tenary search for multi-step sampling on class-conditional ImageNet-64, and LSUN 256
######################################################################################

## CD on ImageNet-64
mpiexec -n 8 python ternary_search.py --begin 0 --end 39 --steps 40 --generator determ --ref_batch /root/consistency/ref_batches/imagenet64.npz --batch_size 256 --model_path /root/consistency/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform

## CT on ImageNet-64
mpiexec -n 8 python ternary_search.py --begin 0 --end 200 --steps 201 --generator determ --ref_batch /root/consistency/ref_batches/imagenet64.npz --batch_size 256 --model_path /root/consistency/ct_imagenet64.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform

## CD on LSUN-256
mpiexec -n 8 python ternary_search.py --begin 0 --end 39 --steps 40 --generator determ-indiv --ref_batch /root/consistency/ref_batches/bedroom256.npz --batch_size 32 --model_path /root/consistency/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform

## CT on LSUN-256
mpiexec -n 8 python ternary_search.py --begin 0 --end 150 --steps 151 --generator determ-indiv --ref_batch /root/consistency/ref_batches/bedroom256.npz --batch_size 32 --model_path /root/consistency/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform

###################################################################
# Multistep sampling on class-conditional ImageNet-64, and LSUN 256
###################################################################

## Two-step sampling for CD (LPIPS) on ImageNet-64
mpiexec -n 8 python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path /path/to/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CD (L2) on ImageNet-64
mpiexec -n 8 python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path /path/to/cd_imagenet64_l2.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CT on ImageNet-64
mpiexec -n 8 python image_sample.py --batch_size 256 --training_mode consistency_training --sampler multistep --ts 0,106,200 --steps 201 --model_path /path/to/ct_imagenet64.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CD (LPIPS) on LSUN-256
mpiexec -n 8 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,17,39 --steps 40 --model_path /path/to/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CD (l2) on LSUN-256
mpiexec -n 8 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,18,39 --steps 40 --model_path /path/to/cd_bedroom256_l2.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CT on LSUN Bedroom-256
mpiexec -n 8 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,67,150 --steps 151 --model_path /path/to/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform

## Two-step sampling for CT on LSUN Cat-256
mpiexec -n 8 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,62,150 --steps 151 --model_path /path/to/ct_cat256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform
