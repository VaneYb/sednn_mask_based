#!/bin/bash

###############
# Export Python3 Path to Environment
export PATH=$PATH:/home/data/yangbeibei/Applications/Anaconda3/bin/
export TF_CPP_MIN_LOG_LEVEL=2

# Dataset directories
TR_SPEECH_DIR="mini_data/train_speech"
TR_NOISE_DIR="mini_data/train_noise"
TE_SPEECH_DIR="mini_data/test_speech"
TE_NOISE_DIR="mini_data/test_noise"
echo "Using mini data. "

# Temporary data directories
WORKSPACE="workspace"
mkdir $WORKSPACE

###############
# 1. Create mixture csv. 
mkdir -p $WORKSPACE
python3.7 -u prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=3
python3.7 prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test


# 2. Calculate mixture features.
SNR=(-5 0 5 10 15 20)
for snr in ${SNR[@]}
do
    TR_SNR=$snr
    python3.7 -u prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
    
    TE_SNR=$snr
    python3.7 prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR
done

# 3. Pack features. 
python3.7 -u prepare_data.py shuffle_and_pack_features --workspace=$WORKSPACE --data_type=train
python3.7 -u prepare_data.py shuffle_and_pack_features --workspace=$WORKSPACE --data_type=test


# 4. Train. 
LEARNING_RATE=1e-3
CUDA_VISIBLE_DEVICES=3,4 python3.7 -u main_dnn.py train --workspace=$WORKSPACE --lr=$LEARNING_RATE


# 5. Inference, enhanced wavs will be created. 
TR_SNR=5
TE_SNR=5
CUDA_VISIBLE_DEVICES=3 python3.7 main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR 


