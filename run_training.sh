#!/bin/bash
# run_training.sh
# Execute this on the high-performance Linux machine.
# Optimized for machines with 32GB+ RAM and an NVIDIA GPU.

echo "Starting Training for Camelyon16..."
configs=("MaxS" "MaxMinS" "AFS" "MAS")

for config in "${configs[@]}"; do
    echo "Running configuration: $config on Camelyon16"
    # Using 4 workers to speed up data loading (requires sufficient RAM)
    python3 Main_DTFD_MIL.py --name "Cam16_$config" --distill_type "$config" --EPOCH 200 --num_workers 4 --device cuda
done

echo "Starting Training for TCGA Lung..."
tcga_configs=("MaxMinS" "AFS")

for config in "${tcga_configs[@]}"; do
    echo "Running configuration: $config on TCGA Lung"
    python3 Main_DTFD_MIL.py --name "TCGA_$config" --distill_type "$config" --EPOCH 200 --num_workers 4 --device cuda \
        --mDATA0_dir_train0 Dataset/TCGA_Lung/mDATA_train_TCGA.pkl \
        --mDATA0_dir_val0 Dataset/TCGA_Lung/mDATA_test_TCGA.pkl \
        --mDATA_dir_test0 Dataset/TCGA_Lung/mDATA_test_TCGA.pkl
done

echo "All training finished!"
