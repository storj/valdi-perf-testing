# Llama3 TPS Measurement Script: Installation and Usage Instructions

```
usage: llama3-perf.py [--num_runs NUM_RUNS] [--multi_model] [--csv_output CSV_OUTPUT] [-h] path

Measure performance for Llama3 models

positional arguments:
  path                  Path to a single model or directory containing multiple models

options:
  --num_runs NUM_RUNS   Number of runs for each measurement
  --multi_model         Process multiple models in the specified directory
  --csv_output CSV_OUTPUT
                        Path to output CSV file
  -h, --help            Show this help message and exit
```

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (required)
- CUDA toolkit and cuDNN installed
- Hugging Face account with access to Llama3 model

## Installation Steps

1. Install the required packages:

   ```

   pip install transformers pynvml psutil huggingface_hub torch accelerate pathlib

   ```

   Note: Adjust the CUDA version (cu118) in the PyTorch installation command to match your system's CUDA version.
2. Set up Hugging Face CLI:

   ```
   huggingface-cli login
   ```

   Follow the prompts to log in with your Hugging Face account credentials.

3. Accept the Llama3.x model license:

   - Visit the Llama3.x model page on Hugging Face and select the model you want to work with(e.g., https://huggingface.co/meta-llama/)
   - Click on "Access repository" and accept the license agreement
4. Download the model using huggingface-cli:

   ```
   huggingface-cli download meta-llama/Meta-Llama-3.1-8B --local-dir ./meta-llama/Meta-Llama-3.1-8B --exclude "original/*"
   ```

   Note: Replace "Meta-Llama-3.1-8B" with the specific Llama3 model version you want to use.

## Usage

1. To see usage information and available options, use the "-h" or "--help" flag:

```
   python llama3-perf.py -h

```

2. Run the script, specifying the path to the downloaded model:

```

   python llama3-perf.py ./meta-llama/Meta-Llama-3.1-8B

```

    or for multiple models:

```

   python llama3-perf.py ./meta-llama --multi_model

```

3. By default, the script will perform 5 runs for TPS measurement. To specify a different number of runs, use the `--num_runs` argument:

```
   
  python llama3-perf.py ./meta-llama/Meta-Llama-3.1-8B --num_runs 10

```

The script will load the model, perform a warm-up run, and then measure the tokens per second (TPS), GPU memory usage, and CPU utilization for the specified number of runs.

The results will be printed to the console, including:

- TPS, GPU memory usage, and CPU utilization for each individual run
- Average TPS, standard deviation of TPS, average GPU memory usage, and average CPU utilization across all runs

## Notes

- This script requires a CUDA-capable GPU. It will not run on CPU.
- The script uses half-precision (FP16) for faster inference.
- A warm-up run is performed before the actual measurements to ensure more accurate results.
- Make sure you have sufficient GPU memory to load the model. If you encounter out-of-memory errors, consider using a smaller model.
- Adjust the `config` in the script to modify parameters such as `max_new_tokens`, `temperature`, etc.
- GPU memory usage is reported in megabytes (MB).
- CPU utilization is reported as a percentage of total CPU capacity.
- Ensure you have proper permissions and have accepted the license agreement for the Llama3 model on Hugging Face.
