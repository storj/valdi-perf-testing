import time
import torch
import gc  # Garbage collection
from transformers import MllamaForConditionalGeneration, AutoProcessor
import psutil  # Library for CPU and memory monitoring
import gpustat  # Library to get GPU stats
import os
import csv

# Initialize model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("Loading model...")
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 to reduce memory usage
    device_map="auto"  # Automatically map to GPU/CPU without explicit memory constraints
)

# Tie weights for model consistency
model.tie_weights()

processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded successfully.")

# Function to monitor resources
def monitor_resources():
    cpu_usage = psutil.cpu_percent(interval=None)
    mem_info = psutil.virtual_memory()

    try:
        gpu_stats = gpustat.new_query()
        gpu_stat = gpu_stats.gpus[0]  # Assuming single GPU
        gpu_usage = gpu_stat.utilization
        gpu_mem_used = gpu_stat.memory_used
        gpu_mem_total = gpu_stat.memory_total
    except Exception as e:
        print(f"Error retrieving GPU stats: {e}")
        gpu_usage = None
        gpu_mem_used = None
        gpu_mem_total = None

    return cpu_usage, mem_info.percent, gpu_usage, gpu_mem_used, gpu_mem_total

# Function to generate a long text prompt
def generate_long_prompt(token_length=100):
    base_sentence = "Could you please provide a detailed explanation about "
    repeat_times = (token_length // len(base_sentence.split())) + 1
    prompt = (base_sentence * repeat_times).strip()
    return prompt[:token_length * 5]  # Adjusting for desired token length

# Process requests sequentially, delete variables after inference

def process_requests_sequentially(requests):
    results = []
    for request in requests:
        user_id = request['user_id']
        prompt = request['prompt']
        arrival_time = request['arrival_time']

        # Record the time when processing starts
        start_time = time.time()

        try:
            # Preprocess and move prompt to the correct device
            inputs = processor(text=prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # Generate model output
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=100, use_cache=False)  # Reduced max tokens to lower memory

            # Record the end time
            end_time = time.time()

            # Calculate timing statistics
            processing_time = end_time - start_time
            waiting_time = start_time - arrival_time
            total_time = end_time - arrival_time

            # Store results
            results.append({
                'user_id': user_id,
                'arrival_time': arrival_time,
                'start_time': start_time,
                'end_time': end_time,
                'waiting_time': waiting_time,
                'processing_time': processing_time,
                'total_time': total_time
            })

        except Exception as e:
            print(f"Error during inference for user {user_id}: {e}")
            if 'CUDA error' in str(e) or 'out of memory' in str(e):
                print("Critical CUDA error encountered. Exiting.")
                break

        finally:
            # Clear GPU memory and trigger garbage collection, but only if inputs and output exist
            if 'inputs' in locals():
                del inputs
            if 'output' in locals():
                del output
            torch.cuda.empty_cache()
            gc.collect()

    return results

# Simulate queue processing
def test_scalability():
    user_counts = [10, 20, 30, 50, 100]  # Different organization sizes
    csv_file = "scalability_results.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Num Users", "Avg Time per User (s)", "Total Time (s)", "CPU Usage (%)",
            "RAM Usage (%)", "GPU Usage (%)", "GPU Memory Used (MB)", "GPU Memory Total (MB)"
        ])

    for num_users in user_counts:
        print(f"Simulating {num_users} users in a queue...")

        # Create fake requests
        arrival_times = [time.time() + i for i in range(num_users)]
        prompts = [generate_long_prompt(token_length=100) for _ in range(num_users)]
        requests = [{'user_id': i, 'prompt': prompts[i], 'arrival_time': arrival_times[i]} for i in range(num_users)]

        start_time = time.time()

        # Monitor system resources before starting
        cpu_usage_start, ram_usage_start, gpu_usage_start, gpu_mem_used_start, _ = monitor_resources()

        # Process all requests
        results = process_requests_sequentially(requests)

        # Monitor system resources after finishing
        cpu_usage_end, ram_usage_end, gpu_usage_end, gpu_mem_used_end, gpu_mem_total = monitor_resources()

        # Average CPU, RAM, and GPU usage during the test
        cpu_usage = (cpu_usage_start + cpu_usage_end) / 2
        ram_usage = (ram_usage_start + ram_usage_end) / 2
        gpu_usage = (gpu_usage_start + gpu_usage_end) / 2 if gpu_usage_start and gpu_usage_end else None
        gpu_mem_used = gpu_mem_used_end

        total_time = time.time() - start_time
        avg_time_per_user = sum([result['total_time'] for result in results]) / num_users if results else 0

        # Write results to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                num_users, avg_time_per_user, total_time, cpu_usage, ram_usage,
                gpu_usage if gpu_usage is not None else "N/A",
                gpu_mem_used if gpu_mem_used is not None else "N/A",
                gpu_mem_total if gpu_mem_total is not None else "N/A"
            ])

        print(f"Completed {num_users} users in {total_time:.2f} seconds with avg time per user {avg_time_per_user:.2f} seconds.\n")

    # Print the success message when the file is saved
    print(f"Results saved to {csv_file}")

# Start the test
if __name__ == "__main__":
    test_scalability()
