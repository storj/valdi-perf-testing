import torch
import time
import csv
from statistics import mean, stdev
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Set up the environment
# Determine the device to run the model on (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that CUDA (GPU acceleration) is available
if device.type != "cuda":
    raise RuntimeError("CUDA is not available. This script requires NVIDIA GPUs.")

# 2. Load and configure the model
# Specify the model name (make sure the model is available or downloaded)
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# Define different batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16]
# Define different maximum input lengths (number of tokens in the input)
MAX_INPUT_LENGTHS = [128, 256, 512, 1024]
# Number of times to repeat each test for averaging
NUM_ITERATIONS = 5

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token to avoid issues with different input lengths
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

# Set the pad token ID in the model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# Tie model weights (if required)
if hasattr(model, "tie_weights"):
    model.tie_weights()

# Set the model to evaluation mode (disables training-specific layers)
model.eval()

# 3. Define the test function
def run_inference(batch_size, max_input_length):
    """
    Measures the tokens-per-second (TPS) for a given batch size and input length.

    -> Parameters:
    batch_size (int): Number of prompts to process simultaneously.
    max_input_length (int): Maximum number of tokens in each input prompt.

    -> Returns:
    avg_tps (float): Average tokens per second over the iterations.
    std_tps (float): Standard deviation of tokens per second.
    """
    tokens_per_second_list = []

    # Repeat the test multiple times for averaging
    for _ in range(NUM_ITERATIONS):
        # Generate realistic prompts by repeating a sample sentence
        sample_sentence = (
            "Act as a Continuous Improvement Analyst. Begin by identifying a problem related to [topic]. "
            "Use the 'Five Whys Technique' to explore the root cause, asking 'why?' five times. "
            "Once the root cause is identified, apply TRIZ principles to devise innovative solutions. "
            "Document the process and outcomes. Repeat the cycle with a new problem or refine the existing solution further."
        )

        # Adjust the prompt to match the desired input length
        prompt = (sample_sentence * ((max_input_length // len(sample_sentence)) + 1))[:max_input_length]
        prompts = [prompt] * batch_size  # Create a list of identical prompts

        # Tokenize the prompts (convert text to token IDs)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Record the start time
        start_time = time.time()

        # Disable gradient calculation for inference
        with torch.no_grad():
            # Generate output tokens from the model
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,  # Reduced to prevent OOM errors
                do_sample=False,  # Keep deterministic behavior
            )

        # Record the end time
        end_time = time.time()

        # Calculate the total number of tokens generated (input + output)
        total_tokens = outputs.numel()

        # Calculate the time taken for inference
        time_taken = end_time - start_time

        # Calculate tokens per second
        tokens_per_second = total_tokens / time_taken

        # Append the TPS to the list
        tokens_per_second_list.append(tokens_per_second)

    # Calculate the average and standard deviation of TPS
    avg_tps = mean(tokens_per_second_list)
    std_tps = stdev(tokens_per_second_list)
    return avg_tps, std_tps

# 4. Run the tests and save results
# Initialize a list to store the results
results = []

# Iterate over each batch size
for batch_size in BATCH_SIZES:
    # Iterate over each input length
    for max_input_length in MAX_INPUT_LENGTHS:
        # Run the inference test
        avg_tps, std_tps = run_inference(batch_size, max_input_length)

        # Store the results in a dictionary
        results.append({
            'Batch Size': batch_size,
            'Input Length': max_input_length,
            'Average TPS': avg_tps,
            'Std Dev TPS': std_tps,
        })

        # Print the results for this configuration
        print(f"Batch Size: {batch_size}, Input Length: {max_input_length}, "
              f"Average TPS: {avg_tps:.2f}, Std Dev: {std_tps:.2f}")

# 5. Save results to CSV
# Specify the output CSV file name
csv_file = 'inference_time_text.csv'

# Get the keys (column names) from the first result entry
keys = results[0].keys()

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as output_file:
    # Create a CSV DictWriter object
    dict_writer = csv.DictWriter(output_file, keys)

    # Write the header (column names)
    dict_writer.writeheader()

    # Write the rows of data
    dict_writer.writerows(results)

print(f"Results saved to {csv_file}")

