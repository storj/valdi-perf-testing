# Llama3 TPS Measurement Script: Installation and Usage Instructions

## Prerequisites
- Python 3.7 or higher
- CUDA-capable GPU (required)
- CUDA toolkit and cuDNN installed
- Hugging Face account with access to Llama3 model

## Installation Steps

1. Create a new virtual environment (recommended):
   ```
   python -m venv llama3_env
   source llama3_env/bin/activate  # On Windows, use: llama3_env\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pynvml psutil huggingface_hub
   ```
   Note: Adjust the CUDA version (cu118) in the PyTorch installation command to match your system's CUDA version.

3. Set up Hugging Face CLI:
   ```
   huggingface-cli login
   ```
   Follow the prompts to log in with your Hugging Face account credentials.

4. Accept the Llama3 model license:
   - Visit the Llama3 model page on Hugging Face (e.g., https://huggingface.co/meta-llama/Llama-3-7b)
   - Click on "Access repository" and accept the license agreement

5. Download the model using huggingface-cli:
   ```
   huggingface-cli download meta-llama/Llama-3-7b --local-dir ./Llama-3-7b --local-dir-use-symlinks False
   ```
   Note: Replace "Llama-3-7b" with the specific Llama3 model version you want to use.

## Usage

1. Ensure your GPU is properly set up and recognized by your system.

2. To see usage information and available options, use the "-h" or "--help" flag:
   ```
   python llama3_tps_measurement.py -h
   ```

3. Run the script, specifying the path to the downloaded model:
   ```
   python llama3_tps_measurement.py ./Llama-3-7b
   ```

4. By default, the script will perform 5 runs for TPS measurement. To specify a different number of runs, use the `--num_runs` argument:
   ```
   python llama3_tps_measurement.py ./Llama-3-7b --num_runs 10
   ```

5. The script will load the model, perform a warm-up run, and then measure the tokens per second (TPS), GPU memory usage, and CPU utilization for the specified number of runs.

6. The results will be printed to the console, including:
   - TPS, GPU memory usage, and CPU utilization for each individual run
   - Average TPS, standard deviation of TPS, average GPU memory usage, and average CPU utilization across all runs

## Notes

- This script requires a CUDA-capable GPU. It will not run on CPU.
- The script uses half-precision (FP16) for faster inference.
- A warm-up run is performed before the actual measurements to ensure more accurate results.
- Make sure you have sufficient GPU memory to load the model. If you encounter out-of-memory errors, consider using a smaller model.
- Adjust the `config` dictionary in the script to modify parameters such as `max_new_tokens`, `temperature`, etc.
- GPU memory usage is reported in megabytes (MB).
- CPU utilization is reported as a percentage of total CPU capacity.
- Ensure you have proper permissions and have accepted the license agreement for the Llama3 model on Hugging Face.
- The `--local-dir-use-symlinks False` option in the download command ensures that actual files are downloaded instead of symlinks, which can be important for some setups.