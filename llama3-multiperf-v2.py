import argparse
import time
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import deepspeed
import pynvml
import psutil
import os
from pathlib import Path

# Try to import DeepSpeed, fall back to DataParallel if not available
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("DeepSpeed not available. Falling back to custom pipeline parallelism.")


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

def get_model_device(model):
    if hasattr(model, 'module'):
        return next(model.module.parameters()).device
    return next(model.parameters()).device

def generate_wrapper(model, **kwargs):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.generate(**kwargs)
    return model.generate(**kwargs)



class PipelineParallelModel:
    def __init__(self, model, num_gpus):
        self.num_gpus = num_gpus
        total_layers = len(list(model.model.layers))
        layers_per_gpu = total_layers // num_gpus
        self.devices = [f'cuda:{i}' for i in range(num_gpus)]
        
        self.dtype = torch.float16  # Set the desired dtype
        
        self.embed = model.model.embed_tokens.to(self.devices[0]).to(self.dtype)
        self.layers = [
            model.model.layers[i * layers_per_gpu : (i + 1) * layers_per_gpu].to(self.devices[i]).to(self.dtype)
            for i in range(num_gpus)
        ]
        self.norm = model.model.norm.to(self.devices[-1]).to(self.dtype)
        self.lm_head = model.lm_head.to(self.devices[-1]).to(self.dtype)
        
        # RoPE-related attributes
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.max_position_embeddings = model.config.max_position_embeddings
        self.rope_theta = model.config.rope_theta
        
    def prepare_attention_mask(self, attention_mask, input_shape):
        # Convert attention_mask to the correct dtype
        attention_mask = attention_mask.to(dtype=self.dtype)
        
        # Create causal mask
        seq_length = input_shape[1]
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=self.dtype, device=attention_mask.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask * torch.finfo(self.dtype).min
        
        # Expand attention_mask
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        expanded_mask = (1.0 - expanded_mask) * torch.finfo(self.dtype).min
        
        # Combine causal mask and attention mask
        combined_mask = expanded_mask + causal_mask
        
        return combined_mask
        
    def rotary_emb(self, x, seq_len):
        device = x.device
        t = torch.arange(seq_len, device=device).type_as(x)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(self.dtype)
        cos = emb.cos()
        sin = emb.sin()
        return cos.unsqueeze(0), sin.unsqueeze(0)
        
    def generate(self, input_ids, attention_mask, **kwargs):
        try:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Prepare attention mask
            attention_mask = self.prepare_attention_mask(attention_mask, input_ids.shape)
            
            # Compute rotary position embeddings
            position_embeddings = self.rotary_emb(torch.ones(1, dtype=self.dtype, device=device), seq_len)
            
            hidden_states = self.embed(input_ids.to(self.devices[0])).to(self.dtype)
            
            for i, device in enumerate(self.devices):
                hidden_states = hidden_states.to(device).to(self.dtype)
                attention_mask = attention_mask.to(device).to(self.dtype)
                position_embeddings = (position_embeddings[0].to(device).to(self.dtype), 
                                       position_embeddings[1].to(device).to(self.dtype))
                for layer in self.layers[i]:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=None,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs[0].to(self.dtype)
            
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            return next_token.unsqueeze(1).to('cuda:0')
        except Exception as e:
            print(f"Error in generate method: {e}")
            print(f"input_ids shape: {input_ids.shape if input_ids is not None else None}")
            print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
            print(f"position_embeddings shape: {position_embeddings[0].shape if 'position_embeddings' in locals() else None}")
            print(f"hidden_states dtype: {hidden_states.dtype if 'hidden_states' in locals() else None}")
            raise


def measure_tps(model, tokenizer, input_text, config):
    try:
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to('cuda:0')
        attention_mask = inputs['attention_mask'].to('cuda:0')
        
        #print(f"Initial input_ids shape: {input_ids.shape}")
        #print(f"Initial attention_mask shape: {attention_mask.shape}")
        
        # Start CPU measurement
        process = psutil.Process()
        cpu_percent_start = process.cpu_percent()
        start_time = time.time()
        
        # Measure initial GPU info
        initial_gpu_info = get_gpu_info()
        
        # Inference
        with torch.no_grad():
            for i in range(config['max_new_tokens']):
                new_token = model.generate(input_ids, attention_mask)
                if new_token is None:
                    print(f"generate returned None at step {i}")
                    break
                input_ids = torch.cat([input_ids, new_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(new_token)], dim=1)
                # print(f"Step {i}: input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
        
        # Measure final GPU info
        final_gpu_info = get_gpu_info()
        
        end_time = time.time()
        
        # End CPU measurement
        cpu_percent_end = process.cpu_percent()
        cpu_percent = (cpu_percent_start + cpu_percent_end) / 2
        
        generated_tokens = input_ids.shape[1] - inputs['input_ids'].shape[1]
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


def run_multiple_measurements(model, tokenizer, input_text, config):
    print("Performing warm-up run...")
    try:
        _ = measure_tps(model, tokenizer, input_text, config)
    except Exception as e:
        print(f"Warm-up run failed: {str(e)}")
        return None, None, None, None, None, None

    print(f"Measuring TPS over {config['num_runs']} runs...")
    results = []
    for i in range(config['num_runs']):
        try:
            tps, gpu_memory, gpu_percent, cpu_percent = measure_tps(model, tokenizer, input_text, config)
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
    

def process_model(model_path, config):
    print(f"\nProcessing model: {model_path}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for inference")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    if HAS_DEEPSPEED:
        try:
            # Initialize DeepSpeed inference engine
            ds_config = {
                "tensor_parallel": {
                    "tp_size": num_gpus
                },
                "dtype": "fp16",
                "injection_policy": {
                    "model_dtype": "fp16",
                    "tensor_parallel": {
                        "enabled": True
                    }
                },
                "replace_method": "auto"
            }
            model = deepspeed.init_inference(base_model, config=ds_config)
            print("Using DeepSpeed for multi-GPU inference")
        except Exception as e:
            print(f"Failed to initialize DeepSpeed: {e}")
            print("Falling back to custom pipeline parallelism")
            model = PipelineParallelModel(base_model, num_gpus)
    else:
        model = PipelineParallelModel(base_model, num_gpus)

    input_text = "In a world where artificial intelligence has become ubiquitous, what are the options for life?"

    try:
        avg_tps, std_dev_tps, avg_gpu_memory, avg_gpu_percent, avg_cpu_percent, results = run_multiple_measurements(model, tokenizer, input_text, config)
        
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
        else:
            print(f"\nFailed to obtain results for {model_path}")
    except Exception as e:
        print(f"Error in process_model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires at least one CUDA-capable GPU.")

    parser = argparse.ArgumentParser(description="Measure performance for Llama3 models using multi-GPU inference", add_help=False)
    parser.add_argument("path", type=str, help="Path to a single model or directory containing multiple models")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for each measurement")
    parser.add_argument("--multi_model", action="store_true", help="Process multiple models in the specified directory")
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                        help="Show this help message and exit")
    args = parser.parse_args()

    config = {
        'min_new_tokens': 100,
        'max_new_tokens': 150,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40,
        'num_return_sequences': 1,
        'num_runs': args.num_runs
    }

    if args.multi_model:
        base_path = Path(args.path)
        if not base_path.is_dir():
            raise ValueError(f"The specified path {args.path} is not a directory.")
        
        model_paths = [p for p in base_path.iterdir() if p.is_dir()]
        if not model_paths:
            raise ValueError(f"No subdirectories found in {args.path}")
        
        for model_path in model_paths:
            process_model(str(model_path), config)
    else:
        process_model(args.path, config)

if __name__ == "__main__":
    main()