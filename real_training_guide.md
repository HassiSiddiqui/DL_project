# Comprehensive Guide: Training DTFD-MIL in Production

This document outlines why training locally on your current machine is unfeasible, details the hardware requirements for a production machine, and provides a step-by-step guide on how to transfer and train your model.

## 1. Why Local Training on this Machine Fails

You currently have **8GB of system RAM** and an AMD RX 6550M GPU. 
- **The Memory Bottleneck:** The Camelyon16 training dataset (`mDATA_train.pkl`) is **11GB**. Just loading this file into Python exceeds your physical memory. Your operating system compensates by using "swap" (using your hard drive as slow RAM), which makes execution astronomically slow.
- **The Threading Problem:** You asked if utilizing more threads (`num_workers > 0`) would help. In PyTorch, using multiple workers for data loading utilizes multiprocessing. This *copies* the dataset into memory for every thread. If you use 4 threads, PyTorch tries to allocate 44GB of RAM (11GB x 4). This will instantly crash your computer with an Out-Of-Memory (OOM) error.
- **CPU vs GPU:** Standard PyTorch on Windows does not support AMD GPUs out-of-the-box (it requires NVIDIA's CUDA). Therefore, your script falls back to CPU training. Training a deep learning Multiple Instance Learning (MIL) model for 200 epochs on a CPU would take months.

**Conclusion:** To get actual research results, you *must* train this on a high-performance machine or a cloud service (like Google Colab Pro, AWS, or RunPod).

---

## 2. Hardware Requirements for the Target Computer

To train this model efficiently, the target machine should have:
1. **System RAM:** Minimum **32GB** (64GB highly recommended). This is required to load the 11GB pickle file and allow PyTorch to use multi-threading (`num_workers=4`) without crashing.
2. **GPU:** An **NVIDIA GPU** with at least **16GB VRAM** (e.g., RTX 3090, RTX 4090, Tesla V100, A100, or a T4 on cloud). PyTorch is highly optimized for NVIDIA GPUs via CUDA.
3. **OS:** Linux (Ubuntu) is the industry standard for deep learning, though Windows with CUDA installed will also work.

---

## 3. Files Required for Transfer

You do not need to transfer everything. Zip the following files and folders from your current directory to transfer to the new computer:

- **Source Code:** `Main_DTFD_MIL.py`, `utils.py`, `requirements.txt`, and the entire `Model/` folder.
- **Data Prep Scripts:** `scratch/prepare_tcga.py`
- **Execution Scripts:** `run_training.sh` (provided below).
- **Datasets:** The entire `Dataset/` folder containing your Camelyon16 `.pkl` files and TCGA_Lung `.zip` files.

---

## 4. Step-by-Step Training Instructions on the New Computer

Assume the new computer is a Linux machine (e.g., an AWS instance or an SSH server).

### Step 1: Transfer and Unzip
Transfer the zipped folder to the new machine and unzip it.
```bash
unzip dtfd_mil_project.zip
cd dtfd_mil_project
```

### Step 2: Set up the Environment
It's best to create a virtual environment. The new machine MUST have NVIDIA CUDA drivers installed.
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (check pytorch.org for exact command based on CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip3 install scikit-learn numpy matplotlib tensorboard tqdm
```

### Step 3: Prepare the TCGA Dataset
If you haven't extracted the TCGA zip files yet, run the preparation script to generate the `.pkl` files.
```bash
python3 scratch/prepare_tcga.py
```

### Step 4: Execute the Training Pipeline
I have created a Linux shell script (`run_training.sh`) that will run the configurations sequentially in the background. It sets `--num_workers 4` to speed up data loading since the new machine will have enough RAM.

Run the script:
```bash
bash run_training.sh
```

### Step 5: Monitor Progress
The script will output progress. You can also monitor the TensorBoard logs to see your loss and AUC graphs in real-time:
```bash
tensorboard --logdir=./debug_log/LOG/
```

### Step 6: Generate Final Results
Once all 200 epochs for all models are complete, transfer the best model weights (saved in `debug_log/`) and the CSV logs back to your computer. You can then plug those real numbers into the report!
