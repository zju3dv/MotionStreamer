NUM_GPUS=${1:-1}  # default: 1 GPU

BATCH_SIZE=$((128 / NUM_GPUS))

echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

accelerate launch --num_processes $NUM_GPUS train_causal_TAE.py \
--batch-size $BATCH_SIZE \
--lr 0.00005 \
--total-iter 2000000 \
--lr-scheduler 1900000 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname t2m_272 \
--vq-act relu \
--exp-name causal_TAE \
--root_loss 7.0 \
--latent_dim 16 \
--hidden_size 1024 \
--num_gpus $NUM_GPUS