NUM_GPUS=${1:-1}  # default: 1 GPU

BATCH_SIZE=$((256 / NUM_GPUS))

echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

accelerate launch --num_processes $NUM_GPUS train_t2m.py \
--batch-size $BATCH_SIZE \
--lr 0.0001 \
--total-iter 100000 \
--out-dir Experiments \
--exp-name t2m_model \
--dataname t2m_272 \
--latent_dir humanml3d_272/t2m_latents \
--num_gpus $NUM_GPUS