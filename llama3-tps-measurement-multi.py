import argparse
import time
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pynvml
import psutil
import os

# Configure PyTorch to use TF32 precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable tensor cores and set memory split
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

pynvml.nvmlInit()

def get_gpu_info():
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_info.append({
            'memory_used': memory_info.used / 1024**2,  # Convert to MB
            'utilization': utilization.gpu
        })
    return gpu_info

def measure_tps(model, tokenizer, input_text, config):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Start CPU measurement
    process = psutil.Process()
    cpu_percent_start = process.cpu_percent()
    start_time = time.time()
    
    # Measure initial GPU info
    initial_gpu_info = get_gpu_info()
    
    # Inference
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config['max_new_tokens'],
            do_sample=config['do_sample'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            top_k=config['top_k'],
            num_return_sequences=config['num_return_sequences'],
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Measure final GPU info
    final_gpu_info = get_gpu_info()
    
    end_time = time.time()
    
    # End CPU measurement
    cpu_percent_end = process.cpu_percent()
    cpu_percent = (cpu_percent_start + cpu_percent_end) / 2
    
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tps = generated_tokens / (end_time - start_time)
    
    # Calculate peak memory and average utilization across all GPUs
    peak_memory = max(max(final['memory_used'] for final in final_gpu_info),
                      max(initial['memory_used'] for initial in initial_gpu_info))
    avg_gpu_percent = statistics.mean(final['utilization'] for final in final_gpu_info)
    
    return tps, peak_memory, avg_gpu_percent, cpu_percent

def run_multiple_measurements(model, tokenizer, input_text, config):
    print("Performing warm-up run...")
    _ = measure_tps(model, tokenizer, input_text, config)
    
    print(f"Measuring TPS over {config['num_runs']} runs...")
    results = []
    for i in range(config['num_runs']):
        tps, gpu_memory, gpu_percent, cpu_percent = measure_tps(model, tokenizer, input_text, config)
        results.append((tps, gpu_memory, gpu_percent, cpu_percent))
        print(f"Run {i+1}/{config['num_runs']}: {tps:.2f} tokens/second, Peak GPU Memory: {gpu_memory:.2f} MB, Avg GPU Utilization: {gpu_percent}%, CPU Utilization: {cpu_percent:.2f}%")
    
    avg_tps = statistics.mean([r[0] for r in results])
    std_dev_tps = statistics.stdev([r[0] for r in results]) if len(results) > 1 else 0
    avg_gpu_memory = statistics.mean([r[1] for r in results])
    avg_gpu_percent = statistics.mean([r[2] for r in results])
    avg_cpu_percent = statistics.mean([r[3] for r in results])
    
    return avg_tps, std_dev_tps, avg_gpu_memory, avg_gpu_percent, avg_cpu_percent, results

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA-capable GPUs.")

    parser = argparse.ArgumentParser(description="Measure TPS for Llama3 model", add_help=False)
    parser.add_argument("model_path", type=str, help="Path to the Llama3 model")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for TPS measurement")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                        help="Show this help message and exit")
    args = parser.parse_args()

    config = {
        'max_new_tokens': 100,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40,
        'num_return_sequences': 1,
        'num_runs': args.num_runs
    }

    print(f"Loading model from {args.model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print(f"Model loaded. Using {torch.cuda.device_count()} GPU(s)")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    input_text = "In a world where artificial intelligence has become ubiquitous,"

    avg_tps, std_dev_tps, avg_gpu_memory, avg_gpu_percent, avg_cpu_percent, results = run_multiple_measurements(model, tokenizer, input_text, config)
    
    print(f"\nResults:")
    print(f"Average tokens per second: {avg_tps:.2f}")
    print(f"Standard deviation of TPS: {std_dev_tps:.2f}")
    print(f"Average Peak GPU Memory Usage: {avg_gpu_memory:.2f} MB")
    print(f"Average GPU Utilization: {avg_gpu_percent:.2f}%")
    print(f"Average CPU Utilization: {avg_cpu_percent:.2f}%")
    print("\nAll runs (tokens/second, Peak GPU Memory MB, Avg GPU %, CPU %):")
    for i, (tps, gpu_memory, gpu_percent, cpu_percent) in enumerate(results, 1):
        print(f"Run {i}: {tps:.2f} TPS, {gpu_memory:.2f} MB, {gpu_percent:.2f}%, {cpu_percent:.2f}%")

if __name__ == "__main__":
    main()
