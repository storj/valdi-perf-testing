import argparse
import time
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pynvml
import psutil
import os
from pathlib import Path
import traceback
import csv

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

def measure_tps(model, tokenizer, config):
    try:
        # Tokenize input text
        inputs = tokenizer(config['input_text'], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        # Start CPU measurement
        process = psutil.Process()
        cpu_percent_start = process.cpu_percent()
        start_time = time.time()
        
        # Measure initial GPU info
        initial_gpu_info = get_gpu_info()
        
        # Inference
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                min_new_tokens=config['min_new_tokens'],
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
    
    except Exception as e:
        print(f"Error in measure_tps: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def run_multiple_measurements(model, tokenizer, config):
    print("Performing warm-up run...")
    try:
        _ = measure_tps(model, tokenizer, config)
    except Exception as e:
        print(f"Warm-up run failed: {str(e)}")
        return None, None, None, None, None, None

    print(f"Measuring TPS over {config['num_runs']} runs...")
    results = []
    for i in range(config['num_runs']):
        try:
            tps, gpu_memory, gpu_percent, cpu_percent = measure_tps(model, tokenizer, config)
            results.append((tps, gpu_memory, gpu_percent, cpu_percent))
            print(f"Run {i+1}/{config['num_runs']}: {tps:.2f} tokens/second, Peak GPU Memory: {gpu_memory:.2f} MB, Avg GPU Utilization: {gpu_percent}%, CPU Utilization: {cpu_percent:.2f}%")
        except Exception as e:
            print(f"Run {i+1} failed: {str(e)}")
            continue

    if not results:
        print("All runs failed. Unable to calculate statistics.")
        return None, None, None, None, None, None

    avg_tps = statistics.mean([r[0] for r in results])
    std_dev_tps = statistics.stdev([r[0] for r in results]) if len(results) > 1 else 0
    avg_gpu_memory = statistics.mean([r[1] for r in results])
    avg_gpu_percent = statistics.mean([r[2] for r in results])
    avg_cpu_percent = statistics.mean([r[3] for r in results])
    
    return avg_tps, std_dev_tps, avg_gpu_memory, avg_gpu_percent, avg_cpu_percent, results

def process_model(model_path, config, csv_writer=None):
    print(f"\nProcessing model: {model_path}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    model.eval()

    try:
        avg_tps, std_dev_tps, avg_gpu_memory, avg_gpu_percent, avg_cpu_percent, results = run_multiple_measurements(model, tokenizer, config)
        
        if avg_tps is not None:
            print(f"\nResults for {model_path}:")
            print(f"Average tokens per second: {avg_tps:.2f}")
            print(f"Standard deviation of TPS: {std_dev_tps:.2f}")
            print(f"Average Peak GPU Memory Usage: {avg_gpu_memory:.2f} MB")
            print(f"Average GPU Utilization: {avg_gpu_percent:.2f}%")
            print(f"Average CPU Utilization: {avg_cpu_percent:.2f}%")
            print("\nAll runs (tokens/second, Peak GPU Memory MB, Avg GPU %, CPU %):")
            for i, (tps, gpu_memory, gpu_percent, cpu_percent) in enumerate(results, 1):
                print(f"Run {i}: {tps:.2f} TPS, {gpu_memory:.2f} MB, {gpu_percent:.2f}%, {cpu_percent:.2f}%")
                
            if csv_writer:
                model_name = Path(model_path).name
                for i, (tps, gpu_memory, gpu_percent, cpu_percent) in enumerate(results, 1):
                    csv_writer.writerow([model_name, i, tps, gpu_memory, gpu_percent, cpu_percent])
        else:
            print(f"\nFailed to obtain results for {model_path}")
    except Exception as e:
        print(f"Error in process_model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one CUDA-capable GPU.")

    parser = argparse.ArgumentParser(description="Measure performance for Llama3 models", add_help=False)
    parser.add_argument("path", type=str, help="Path to a single model or directory containing multiple models")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for each measurement")
    parser.add_argument("--multi_model", action="store_true", help="Process multiple models in the specified directory")
    parser.add_argument("--csv_output", type=str, help="Path to output CSV file")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                        help="Show this help message and exit")
    args = parser.parse_args()

    config = {
        'input_text': "In a world where artificial intelligence has become ubiquitous, what are the options for life?",
        'min_new_tokens': 100,
        'max_new_tokens': 150,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40,
        'num_return_sequences': 1,
        'num_runs': args.num_runs
    }

    csv_file = None
    csv_writer = None
    if args.csv_output:
        csv_file = open(args.csv_output, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Model', 'Run', 'TPS', 'Peak GPU Memory (MB)', 'Avg GPU Utilization (%)', 'CPU Utilization (%)'])

    try:
        if args.multi_model:
            base_path = Path(args.path)
            if not base_path.is_dir():
                raise ValueError(f"The specified path {args.path} is not a directory.")
            
            model_paths = [p for p in base_path.iterdir() if p.is_dir()]
            if not model_paths:
                raise ValueError(f"No subdirectories found in {args.path}")
            
            for model_path in model_paths:
                process_model(str(model_path), config, csv_writer)
        else:
            process_model(args.path, config, csv_writer)
    finally:
        if csv_file:
            csv_file.close()

if __name__ == "__main__":
    main()