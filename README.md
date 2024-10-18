# README.md

## Performance Testing of Llama-3.2-11B Vision-Instruct Model

This guide provides a comprehensive walkthrough for setting up an environment to run performance tests on the **Llama-3.2-11B Vision-Instruct** model developed by Meta. You'll find detailed instructions on:

- Hardware and software configurations
- Setting up the environment
- Running the provided test scripts
- Collecting and downloading results

By the end of this guide, you'll be able to reproduce the tests, analyze the performance, and adapt the setup for your own experiments.

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Requirements](#software-requirements)
- [Setting Up the Environment](#setting-up-the-environment)
  - [1. Accessing the Virtual Machine](#1-accessing-the-virtual-machine)
  - [2. System Update and Essential Dependencies](#2-system-update-and-essential-dependencies)
  - [3. Python Environment Setup](#3-python-environment-setup)
  - [4. Installing Required Python Packages](#4-installing-required-python-packages)
  - [5. Downloading Llama-3.2-11B Vision-Instruct Model](#5-downloading-llama-32-11b-vision-instruct-model)
- [Running the Tests](#running-the-tests)
  - [Test 1: Inference Time with Varying Batch Sizes and Input Lengths](#test-1-inference-time-with-varying-batch-sizes-and-input-lengths)
  - [Test 2: Image Processing Performance](#test-2-image-processing-performance)
  - [Test 4: Scalability Test with Sequential Request Processing](#test-4-scalability-test-with-sequential-request-processing)
- [Collecting and Downloading Results](#collecting-and-downloading-results)
  - [Downloading Result Files](#downloading-result-files)
  - [Uploading Files to the Virtual Machine](#uploading-files-to-the-virtual-machine)
- [Additional Notes](#additional-notes)
- [Conclusion](#conclusion)

---

## Introduction

This guide aims to help AI enthusiasts and small business CTOs replicate performance tests on the **Llama-3.2-11B Vision-Instruct** model. By following this guide, you'll understand:

- How to set up a suitable environment for testing
- How to run various performance tests
- How to collect and interpret the results

---

## Prerequisites

### Hardware Requirements

To run these tests efficiently, you'll need access to a machine with the following specifications:

- **GPU**: NVIDIA GeForce RTX 4090 with 24 GB VRAM
- **CPU**: 20 vCPUs (e.g., AMD EPYC 7B13)
- **RAM**: At least 75 GB
- **Storage**: At least 350 GB SSD

You can rent such a machine from [Valdi.ai](https://valdi.ai/), a platform that allows you to access high-performance GPUs in the cloud.

### Software Requirements

- **Operating System**: Ubuntu 20.04 or later
- **Python**: Version 3.10
- **CUDA**: Version compatible with your GPU drivers (CUDA 11.8 recommended)
- **NVIDIA Drivers**: Latest drivers compatible with your GPU

---

## Setting Up the Environment

### 1. Accessing the Virtual Machine

**Note**: Replace `[your_pem_file.pem]`, `[port_number]`, and `[server_ip]` with your actual `.pem` file name, port number, and server IP address.

1. **Set Permissions for Your PEM File**:

   ```bash
   chmod 400 [your_pem_file].pem
   ```

2. **Connect to the Virtual Machine via SSH**:

   ```bash
   ssh -p [port_number] -i [your_pem_file].pem user@[server_ip]
   ```

   **Example**:

   ```bash
   ssh -p 10034 -i Dima_Levin.pem user@8.17.147.159
   ```

### 2. System Update and Essential Dependencies

Once connected to the VM:

1. **Update and Upgrade the System**:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Essential Dependencies**:

   ```bash
   sudo apt install build-essential git curl wget -y
   ```

### 3. Python Environment Setup

1. **Install Python 3.10 and pip**:

   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip -y
   ```

2. **Create a Virtual Environment**:

   ```bash
   python3.10 -m venv llama3_env
   ```

3. **Activate the Virtual Environment**:

   ```bash
   source llama3_env/bin/activate
   ```

4. **Upgrade pip**:

   ```bash
   pip install --upgrade pip
   ```

### 4. Installing Required Python Packages

1. **Install PyTorch with CUDA Support**:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Hugging Face Transformers and Other Dependencies**:

   ```bash
   pip install transformers huggingface_hub pynvml accelerate requests Pillow
   ```

3. **Install xFormers for Memory Optimization**:

   ```bash
   pip install xformers
   ```

4. **Install Additional Packages for Performance Monitoring**:

   ```bash
   pip install psutil gpustat pandas
   ```

### 5. Downloading Llama-3.2-11B Vision-Instruct Model

**Important**: Ensure you have access to the model on Hugging Face. You **must request access** via the [model page](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) before proceeding.

1. **Login to Hugging Face CLI**:

   ```bash
   huggingface-cli login
   ```

   Follow the prompts to authenticate.

2. **Download the Model**:

   ```bash
   mkdir llama3_models
   huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --local-dir llama3_models --include "model.safetensors*"
   ```

---

## Running the Tests

### Test 1: Inference Time with Varying Batch Sizes and Input Lengths

**Objective**: Measure tokens-per-second (TPS) for different batch sizes and input lengths.

1. **Navigate to the Test Directory**:

   ```bash
   cd ~
   mkdir performance_test
   cd performance_test
   ```

2. **Create the Test Script**:

   To create a new file on your VM, use:

   ```bash
   nano test_1.py
   ```

   Paste the following code into `test_1.py`:

   ```python
   [Paste the code from test_1 here]
   ```

   Save and exit by pressing `Ctrl + O`, then `Enter`, and `Ctrl + X`.

3. **Run the Test**:

   ```bash
   python test_1.py
   ```

4. **Results**:

   The script will output results to the console and save them to `inference_time_text.csv`.

### Test 2: Image Processing Performance

**Objective**: Evaluate the model's image processing capabilities.

1. **Create an Images Directory**:

   ```bash
   mkdir ~/images
   ```

2. **Upload Images to the VM**:

   From your local machine, upload images using `scp`:

   ```bash
   scp -P [port_number] -i [your_pem_file].pem -r /path/to/your/images/* user@[server_ip]:~/images/
   ```

   **Example**:

   ```bash
   scp -P 20013 -i Dima_Levin.pem -r ~/Downloads/img/* user@69.55.141.236:~/images/
   ```

3. **Create the Test Script**:

   In the `performance_test` directory, create `test_2.py`:

   ```bash
   nano test_2.py
   ```

   Paste the following content into `test_2.py`:

   ```python
   [Paste the code from test_2 here]
   ```

   Save and exit (`Ctrl + O`, `Enter`, `Ctrl + X`).

4. **Run the Test**:

   ```bash
   python test_2.py
   ```

5. **Results**:

   The script will process each image and save the results to `image_processing_results.csv`.

### Test 4: Scalability Test with Sequential Request Processing

**Objective**: Assess how the model handles sequential processing of multiple requests.

1. **Create the Test Script**:

   In the `performance_test` directory, create `test_4.py`:

   ```bash
   nano test_4.py
   ```

   Paste the following content into `test_4.py`:

   ```python
   [Paste the code from test_4 here]
   ```

   Save and exit (`Ctrl + O`, `Enter`, `Ctrl + X`).

2. **Run the Test**:

   ```bash
   python test_4.py
   ```

3. **Results**:

   The script will simulate processing for different numbers of users and save the results to `scalability_results.csv`.

---

## Collecting and Downloading Results

After running the tests, you'll want to download the result files to your local machine for analysis.

### Downloading Result Files

From your local machine, use `scp` to download the files:

1. **Download `inference_time_text.csv`**:

   ```bash
   scp -P [port_number] -i [your_pem_file].pem user@[server_ip]:~/performance_test/inference_time_text.csv ~/Downloads/
   ```

2. **Download `image_processing_results.csv`**:

   ```bash
   scp -P [port_number] -i [your_pem_file].pem user@[server_ip]:~/performance_test/image_processing_results.csv ~/Downloads/
   ```

3. **Download `scalability_results.csv`**:

   ```bash
   scp -P [port_number] -i [your_pem_file].pem user@[server_ip]:~/performance_test/scalability_results.csv ~/Downloads/
   ```

**Example**:

```bash
scp -P 20013 -i Dima_Levin.pem user@69.55.141.236:~/performance_test/inference_time_text.csv ~/Downloads/
```

### Uploading Files to the Virtual Machine

If you need to upload additional files to the VM:

1. **Upload Files Using `scp`**:

   ```bash
   scp -P [port_number] -i [your_pem_file].pem /local/path/to/file user@[server_ip]:/remote/path/
   ```

**Example**:

```bash
scp -P 20013 -i Dima_Levin.pem -r ~/Downloads/img/* user@69.55.141.236:~/images/
```

---

## Additional Notes

- **Activating the Environment**:

  Each time you log into the VM, remember to activate your Python virtual environment:

  ```bash
  source ~/llama3_env/bin/activate
  ```

- **Creating and Editing Files**:

  - **Create a New File**:

    Use `nano` to create or edit files:

    ```bash
    nano filename.py
    ```

    - To save changes, press `Ctrl + O`, then `Enter`.
    - To exit `nano`, press `Ctrl + X`.

  - **Delete File Contents Quickly**:

    If you want to delete the content of a file quickly:

    ```bash
    echo '' > filename.py
    ```

- **GPU and CPU Monitoring**:

  - **GPU Monitoring**:

    Run `nvidia-smi` to monitor GPU usage:

    ```bash
    watch -n 1 nvidia-smi
    ```

  - **CPU Monitoring**:

    Use `htop` to monitor CPU and memory:

    ```bash
    htop
    ```

- **CUDA Memory Configuration**:

  Set the following environment variable to optimize CUDA memory allocation:

  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

- **SSH Key Permissions**:

  Ensure your `.pem` file has the correct permissions:

  ```bash
  chmod 400 [your_pem_file].pem
  ```

- **Access to Llama-3.2-11B Vision-Instruct Model**:

  Before downloading the model, you **must request access** on Hugging Face using this link:

  [https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

---

## Conclusion

By following this guide, you've set up an environment to run performance tests on the **Llama-3.2-11B Vision-Instruct** model. You've learned how to:

- Configure the necessary hardware and software
- Run various performance tests
- Collect and download results for analysis

These tests help in understanding the model's capabilities and limitations, especially in terms of inference time, resource utilization, and scalability. Feel free to modify the test scripts and experiment with different configurations to further explore the model's performance.

---

**Next Steps**:

- **Analyze the Results**: Use tools like pandas or Excel to analyze the CSV files and visualize the performance metrics.
- **Optimize the Model**: Experiment with different settings in the test scripts to optimize performance for your specific use case.
- **Scale Up**: Consider running tests on machines with multiple GPUs to assess scalability in a distributed environment.
- **Feedback and Collaboration**: If you have suggestions or improvements, feel free to contribute to the project repository or reach out for collaboration opportunities.

**Happy Testing!**