import onnxruntime as ort
import numpy as np
import time
import os
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def visualize_embedding_heatmap(embedding_vector, title="Embedding Heatmap", save_dir="embedding_plots", filename_prefix="embedding_heatmap", vmin=None, vmax=None, cmap='viridis'):
    """
    Visualizes a 1D embedding vector as a 2D heatmap.

    Args:
        embedding_vector (np.ndarray): 1D NumPy array (e.g., shape (768,)).
        title (str): Title for the plot.
        save_dir (str): Directory to save the plot.
        filename_prefix (str): Prefix for the saved filename.
        vmin (float, optional): Minimum value for the color scale. If None, uses data min.
        vmax (float, optional): Maximum value for the color scale. If None, uses data max.
        cmap (str): Matplotlib colormap name (e.g., 'viridis', 'plasma', 'coolwarm').
    """
    if not isinstance(embedding_vector, np.ndarray):
        print("Error: Input must be a NumPy array.")
        return

    # Ensure the vector is 1D
    embedding_vector = embedding_vector.flatten()
    dim = embedding_vector.shape[0]

    if embedding_vector.ndim != 1:
        print(f"Error: Flattened embedding vector should be 1D, but got shape {embedding_vector.shape}")
        return

    # Reshape to 2D for visualization
    target_height, target_width = 24, 32
    if dim == target_height * target_width:
        shape_2d = (target_height, target_width)
        print(f"Visualizing '{title}' (dimension: {dim}, reshaped to: {shape_2d})")
    else:
        print(f"Warning: Input dimension is {dim}, not {target_height * target_width}. Visualizing as 1x{dim}.")
        shape_2d = (1, dim)
        
    try:
        reshaped_embedding = embedding_vector.reshape(shape_2d)
    except ValueError as e:
        print(f"Error: Could not reshape dimension {dim} to {shape_2d}. {e}")
        return

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Create heatmap
    plt.figure(figsize=(target_width / 3, target_height / 3))
    im = plt.imshow(reshaped_embedding, cmap=cmap, aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    plt.title(title, fontsize=10)
    plt.xticks([])
    plt.yticks([])
    
    # Add colorbar
    plt.colorbar(im, fraction=0.046, pad=0.04)

    heatmap_path = os.path.join(save_dir, f"{filename_prefix}.png")
    try:
        plt.savefig(heatmap_path, dpi=150)
        print(f"Heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    
    plt.close()

# Configuration
output_dir = "/home/orangepi/learnLM/emb_model/"
base_name = "granite_automoel_cls_norm_b1_len64"
cpu_model_path = os.path.join(output_dir, f"{base_name}_cpu.onnx")
npu_model_path = os.path.join(output_dir, f"{base_name}_npu.onnx")
model_path = "ibm-granite/granite-embedding-278m-multilingual"

# Prepare dummy input data
batch_size = 1
seq_length = 64
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenized_np = tokenizer(
    ["This is a test"],
    padding='max_length',
    truncation=True,
    max_length=seq_length,
    return_tensors='np'
)
dummy_input_ids = tokenized_np['input_ids']
dummy_attention_mask = tokenized_np['attention_mask']

# Load models
print("Loading models...")
try:
    cpu_session_options = ort.SessionOptions()
    cpu_session = ort.InferenceSession(cpu_model_path, sess_options=cpu_session_options, providers=['CPUExecutionProvider'])
    print(f"Loaded CPU part: {cpu_model_path}")

    npu_session_options = ort.SessionOptions()
    npu_session = ort.InferenceSession(npu_model_path, sess_options=npu_session_options, providers=['CPUExecutionProvider'])
    print(f"Loaded NPU part (running on CPU for now): {npu_model_path}")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Retrieve model input/output names
cpu_input_names = [inp.name for inp in cpu_session.get_inputs()]
cpu_output_names = [out.name for out in cpu_session.get_outputs()]
npu_input_names = [inp.name for inp in npu_session.get_inputs()]
npu_output_names = [out.name for out in npu_session.get_outputs()]

print("\n--- Model Info ---")
print(f"CPU Model Inputs: {cpu_input_names}")
print(f"CPU Model Outputs (Intermediate): {cpu_output_names}")
print(f"NPU Model Inputs: {npu_input_names}")
print(f"NPU Model Outputs (Final): {npu_output_names}")
print("--------------------\n")

# Prepare CPU part inputs
cpu_input_feed = {}
if 'input_ids' in cpu_input_names:
    cpu_input_feed['input_ids'] = dummy_input_ids
if 'attention_mask' in cpu_input_names:
    cpu_input_feed['attention_mask'] = dummy_attention_mask

# Validate CPU inputs
prepared_cpu_inputs = list(cpu_input_feed.keys())
if set(prepared_cpu_inputs) != set(cpu_input_names):
    print(f"Warning: Mismatch between expected CPU inputs {cpu_input_names} and prepared inputs {prepared_cpu_inputs}.")

# Run CPU part and measure time
print("Running CPU part...")
start_time_cpu = time.perf_counter()
try:
    cpu_outputs = cpu_session.run(cpu_output_names, cpu_input_feed)
except Exception as e:
    print(f"Error running CPU part: {e}")
    exit(1)
cpu_time = (time.perf_counter() - start_time_cpu) * 1000
print(f"CPU part finished in {cpu_time:.4f} ms")

# Prepare NPU part inputs
npu_input_feed = {}
cpu_results_dict = dict(zip(cpu_output_names, cpu_outputs))
for name in npu_input_names:
    if name in cpu_results_dict:
        npu_input_feed[name] = cpu_results_dict[name]
        print(f"  Mapping CPU output '{name}' to NPU input.")

# Add passthrough inputs
passthrough_input_name = 'attention_mask'
if passthrough_input_name in npu_input_names and passthrough_input_name not in npu_input_feed:
    npu_input_feed[passthrough_input_name] = dummy_attention_mask
    print(f"  Passing through original input '{passthrough_input_name}' to NPU input.")

# Validate NPU inputs
prepared_npu_inputs = list(npu_input_feed.keys())
missing_npu_inputs = set(npu_input_names) - set(prepared_npu_inputs)
if missing_npu_inputs:
    print(f"\nError: Could not prepare all required inputs for the NPU part.")
    print(f"  Expected NPU inputs: {npu_input_names}")
    print(f"  Prepared NPU inputs: {prepared_npu_inputs}")
    print(f"  Missing inputs: {missing_npu_inputs}")
    exit(1)
else:
    print("All NPU inputs prepared.")

# Run NPU part and measure time
print("\nRunning NPU part (on CPU)...")
start_time_npu = time.perf_counter()
try:
    npu_outputs = npu_session.run(npu_output_names, npu_input_feed)
except Exception as e:
    print(f"Error running NPU part: {e}")
    exit(1)
npu_time = (time.perf_counter() - start_time_npu) * 1000
print(f"NPU part finished in {npu_time:.4f} ms")

# Calculate total execution time
total_time = cpu_time + npu_time
print(f"\nTotal execution time (CPU + NPU on CPU): {total_time:.4f} ms")

# Process final output
final_output = npu_outputs[0]
print(f"\nFinal output shape: {final_output.shape}")
print(f"Final output type: {final_output.dtype}")
print("Final output (first few elements):", final_output.flatten()[:10])

# Visualize embedding
visualize_embedding_heatmap(final_output.flatten())

# Load and run original model for comparison
start_time = time.time()
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Run model inference
output = model(
    input_ids=torch.LongTensor(dummy_input_ids),
    attention_mask=torch.LongTensor(dummy_attention_mask),
    return_dict=True
).last_hidden_state

# Normalize and print output
print(F.normalize(output[:, 0], p=2, dim=1)[-1, :10])
