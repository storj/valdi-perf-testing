import os
import time
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import csv

# Initialize the model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
print(f"Loading model {model_id}...")
# Load the model with half precision for efficiency, and map it to the appropriate device (GPU if available)
model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
# Load the processor to handle multimodal inputs (images and text)
processor = AutoProcessor.from_pretrained(model_id)
print(f"Model loaded successfully.")

# Directory where images are stored
image_dir = "./images"

# Prepare prompt template to be used with each image
prompt_template = "Can you explain to me as to a blind person what do you see on the image?"

# CSV output file to save the results
output_file = "image_processing_results.csv"

# Check if the output CSV file already exists and set the write mode accordingly
file_exists = os.path.isfile(output_file)

# Open the CSV file for writing results (append if it already exists)
with open(output_file, mode='a' if file_exists else 'w', newline='') as csvfile:
    # Define fieldnames for the CSV
    fieldnames = ['Image', 'Prompt', 'Output', 'TimeTaken']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header only if the file is being created for the first time
    if not file_exists:
        writer.writeheader()

    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        # Only process files with supported image extensions
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, image_file)
            print(f"Processing image: {image_file}")

            # Open and preprocess the image
            image = Image.open(image_path)
            # Create a prompt that combines the image with the text prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_template}
                ]}
            ]

            # Prepare inputs using the processor to format the image and text appropriately for the model
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

            # Measure time taken for model inference
            start_time = time.time()
            # Generate output tokens using the model
            output = model.generate(**inputs, max_new_tokens=30)
            end_time = time.time()

            # Calculate time taken for inference
            time_taken = end_time - start_time

            # Decode the model output into human-readable text
            output_text = processor.decode(output[0], skip_special_tokens=True)

            # Print process information to the console
            print(f"Image: {image_file}, Time Taken: {time_taken:.2f} seconds, Output: {output_text}")

            # Write the results to the CSV file
            writer.writerow({
                'Image': image_file,
                'Prompt': prompt_template,
                'Output': output_text,
                'TimeTaken': f"{time_taken:.2f}"
            })

print(f"Image processing completed. Results saved to {output_file}.")