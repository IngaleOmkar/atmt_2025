#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=12:0:0     
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_inference.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE sk -> en
python translate.py \
    --cuda \
    --input ./en-sk/processed/aligned_en.sk \
    --src-tokenizer ./cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ./cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ./cz-en/checkpoints/checkpoint_best.pt \
    --output ./en-sk/output.txt \
    --max-len 300 \
    --bleu \
    --reference ./en-sk/processed/aligned_sk.en

