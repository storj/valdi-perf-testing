Here is a step-by-step guide to set up your environment on the new virtual machine and get ready to run the tests for Llama 3.2:


### Step 1: Initial Setup

1. **Connect to the Virtual Machine**:

Run this in your terminal to access the VM:

```bash

ssh -p 10034 -i Dima_Levin.pem user@8.17.147.159

```

  

2. **Update and Upgrade the System**:

First, make sure your system is up-to-date:

```bash

sudo apt update && sudo apt upgrade -y≈

```

  

3. **Install Essential Dependencies**:

Install the basic build tools and system dependencies:

```bash

sudo apt install build-essential git curl wget -y

```

  

### Step 2: Install Python and Create Environment

1. **Install Python 3.10**:

Install Python 3.10 if it's not already available on the system:

```bash

sudo apt install python3.10 python3.10-venv python3.10-dev -y

```

  

2. **Install pip**:

Get pip, if it’s not available:

```bash

sudo apt install python3-pip -y

```

  

3. **Create Virtual Environment**:

Create a virtual environment for Python:

```bash

python3.10 -m venv llama3_env

```

  

4. **Activate the Virtual Environment**:

Activate the environment:

```bash

source llama3_env/bin/activate

```

  

5. **Upgrade pip**:

Upgrade pip inside the environment:

```bash

pip install --upgrade pip

```

  

### Step 3: Install Required Libraries

1. **Install PyTorch with CUDA**:

Install PyTorch with CUDA support:

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

  

2. **Install Additional Required Libraries**:

Install `transformers`, `huggingface_hub`, and other libraries:

```bash

pip install transformers huggingface_hub pynvml accelerate requests Pillow

```

  

3. **Install xFormers**:

To optimize memory during multi-modal model testing, install xFormers:

```bash

pip install xformers

```

  

### Step 4: Download and Setup Llama-3.2-11B-Vision-Instruct

| ! NOTE: be sure that you have gor the access to the Hugging Face and to the LLama 3.2 11B model there. For doing this follow this link -> https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct 

1. **Login to Hugging Face CLI**:

You need to authenticate with Hugging Face to download the model:

```bash

huggingface-cli login

```

  

Follow the instructions to log in with your Hugging Face account.

  

2. **Download Llama-3.2-11B-Vision-Instruct**:

Use the Hugging Face CLI to download the model:

```bash

huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct --local-dir llama3_models --include "model.safetensors*"

```

  

### Step 5: Prepare Testing Scripts and Dependencies

1. **Clone the Testing Repository**:

If you have a repository or scripts, clone or copy them to your machine. If not, create a working directory for the performance tests:

```bash

mkdir performance_test && cd performance_test

```

  

2. **Download or Create Performance Testing Script**:

You can use the script that was working for you earlier. Here's a sample command:

```bash

wget https://your-testing-script-link/llama3_perf_vllm.py

```

  

3. **Install Additional Packages for Performance Monitoring**:

Install any performance monitoring packages like `psutil`:

```bash

pip install psutil pandas

```

  

### Step 6: Run the Tests

1. **Ensure the Correct Environment Variables**:

Set CUDA memory configurations:

```bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

```

  

2. **Run the First Test (Tokens per Second)**:

Run the script:

```bash

python llama3_perf_vllm.py

```

  

3. **Placeholder for Image Testing**:

For image-based tests, make sure you have the image files available in a directory. You can upload the images using `scp` or download them with `wget`:

```bash

mkdir images && cd images

wget <image-url-1>

wget <image-url-2>

wget <image-url-3>

```

  

### Step 7: Review and Output Results

1. **Output Results to CSV**:

Ensure the script outputs the results in CSV format. If needed, modify your Python script to append results to a CSV file, like:

```python

import csv

with open("performance_results.csv", "w", newline="") as csvfile:

writer = csv.writer(csvfile)

writer.writerow(["Metric", "Value"])

writer.writerow(["Tokens per Second", avg_tokens_per_second])

```

  

### Step 8: Scaling and Resource Monitoring

1. **Use nvidia-smi for GPU Monitoring**:

Open another terminal and run:

```bash

nvidia-smi

```

  

2. **Use `htop` for CPU Monitoring**:

Run:

```bash

htop

```

  

After completing these steps, you'll have the environment fully set up, ready to run the performance tests for Llama 3.2, and collect results in a structured way. Let me know when you’re ready to move forward with specific tests!