import vllm
import torch
import time
import csv
import random
import string
from statistics import mean, stdev
from vllm import SamplingParams

# 1. Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA is not available. This script requires NVIDIA GPUs.")

# 2. Load and configure the model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = 64
MAX_INPUT_LENGTH = 1000
NUM_ITERATIONS = 6
OUTPUT_FORMAT = "console"

# Configuration options for memory management
GPU_MEMORY_UTILIZATION = 0.8  # Adjust this value between 0 and 1
MAX_NUM_BATCHED_TOKENS = max(BATCH_SIZE * MAX_INPUT_LENGTH, 2048)  # Ensure this is larger than BATCH_SIZE
MAX_NUM_SEQS = BATCH_SIZE  # Set this to be equal to or less than BATCH_SIZE

try:
    model = vllm.LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,  # For future distributed inference, increase this value
        swap_space=0,  # Disable swap space to avoid CPU memory issues
        cpu_offload_gb=50
    )
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# 3. Prepare input data
def generate_random_prompt(length):
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))

# 4. Create an inference loop
def run_inference():
    tokens_per_second_list = []
    sampling_params = SamplingParams(max_tokens=100)
    
    for i in range(NUM_ITERATIONS):
        prompts = [generate_random_prompt(random.randint(10, MAX_INPUT_LENGTH)) for _ in range(BATCH_SIZE)]
        
        start_time = time.time()
        try:
            outputs = model.generate(prompts, sampling_params)
        except Exception as e:
            print(f"Inference failed on iteration {i}: {e}")
            continue
        end_time = time.time()
        
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        time_taken = end_time - start_time
        tokens_per_second = total_tokens / time_taken
        tokens_per_second_list.append(tokens_per_second)
        
        print(f"Iteration {i + 1}: {tokens_per_second:.2f} tokens/second")
    
    return tokens_per_second_list


# 5. Calculate performance metrics
tokens_per_second_results = run_inference()
avg_tokens_per_second = mean(tokens_per_second_results)
std_dev_tokens_per_second = stdev(tokens_per_second_results) if len(tokens_per_second_results) > 1 else 0

# 6. Format and save results
results = {
    "Model": MODEL_NAME,
    "Batch Size": BATCH_SIZE,
    "Max Input Length": MAX_INPUT_LENGTH,
    "Avg Tokens/Second": f"{avg_tokens_per_second:.2f}",
    "Std Dev Tokens/Second": f"{std_dev_tokens_per_second:.2f}"
}

if OUTPUT_FORMAT == "console":
    print("\nPerformance Test Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
elif OUTPUT_FORMAT == "csv":
    with open("performance_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print("Results saved to performance_results.csv")
else:
    print("Invalid output format specified.")

# 7. Comments for future distributed inference implementation
"""
For distributed inference across multiple GPUs:
1. Modify the vllm.LLM initialization:
   - Increase tensor_parallel_size to match the number of GPUs
   - Set gpu_memory_utilization for each GPU

2. Use vllm's AsyncLLMEngine for better performance in distributed setups:
   from vllm import AsyncLLMEngine
   
   engine = AsyncLLMEngine.from_engine_args(
       engine_args=EngineArgs(
           model=MODEL_NAME,
           tensor_parallel_size=num_gpus,
           ...
       )
   )

3. Modify the inference loop to use the AsyncLLMEngine:
   async def generate_async(engine, prompts):
       results = await engine.generate(prompts, ...)
       return results

4. Use asyncio to manage the asynchronous operations:
   import asyncio
   
   async def run_distributed_inference():
       # Similar structure to the current run_inference function,
       # but using asyncio.gather for parallel processing
       ...

   asyncio.run(run_distributed_inference())

5. Consider implementing dynamic batching for more efficient GPU utilization
"""

if __name__ == "__main__":
    print("Performance test completed.")