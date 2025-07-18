import onnx
import os

# Configuration
original_onnx_path = "granite_automoel_cls_norm_b1_len64.onnx"
split_node_name = "/transformer/embeddings/Add_2"  # Node to split model; CPU before, NPU from this node
output_dir = os.path.dirname(original_onnx_path)
base_name = os.path.splitext(os.path.basename(original_onnx_path))[0]
cpu_model_path = os.path.join(output_dir, f"{base_name}_cpu.onnx")
npu_model_path = os.path.join(output_dir, f"{base_name}_npu.onnx")
passthrough_input_names = ['attention_mask']  # Original model inputs required by NPU part

# Load the original model
print(f"Loading original model: {original_onnx_path}")
try:
    original_model = onnx.load(original_onnx_path)
    onnx.checker.check_model(original_model)  # Validate model integrity
    print("Original model loaded and checked successfully.")
except Exception as e:
    print(f"Error loading or checking original model: {e}")
    exit(1)

# Find the split node and retrieve its input tensor names
split_input_tensor_names = None
target_node_found = False
for node in original_model.graph.node:
    if node.name == split_node_name:
        split_input_tensor_names = list(node.input)  # Get input tensors of the split node
        target_node_found = True
        print(f"Found split node '{split_node_name}'.")
        print(f"Input tensors to this node (boundary tensors from CPU to NPU): {split_input_tensor_names}")
        break

if not target_node_found:
    print(f"Error: Split node '{split_node_name}' not found in the model graph.")
    print("\nAvailable node names (first 100):")
    for i, node in enumerate(original_model.graph.node):
        print(f"- {node.name}")
        if i >= 100:  # Limit output to avoid excessive logging
            print("... (list truncated)")
            break
    exit(1)

if not split_input_tensor_names:
    print(f"Error: Split node '{split_node_name}' found, but has no input tensors.")
    exit(1)

# Retrieve original model's input and output names
original_input_names = [inp.name for inp in original_model.graph.input]
original_output_names = [out.name for out in original_model.graph.output]
print(f"Original model inputs: {original_input_names}")
print(f"Original model outputs: {original_output_names}")

# Verify passthrough input names exist in the original model
missing_passthrough = [name for name in passthrough_input_names if name not in original_input_names]
if missing_passthrough:
    print(f"Error: Specified passthrough inputs {missing_passthrough} not found in original model inputs {original_input_names}.")
    exit(1)

# Extract CPU part
# Inputs: Original model inputs
# Outputs: Input tensors of the split node
print(f"\nExtracting CPU part...")
print(f"  Inputs: {original_input_names}")
print(f"  Outputs: {split_input_tensor_names}")
try:
    onnx.utils.extract_model(
        original_onnx_path,
        cpu_model_path,
        input_names=original_input_names,
        output_names=split_input_tensor_names,
        check_model=True  # Validate extracted model
    )
    print(f"CPU part saved to: {cpu_model_path}")
except Exception as e:
    print(f"Error extracting or checking CPU part: {e}")
    print("Ensure all tensor names in 'Outputs' exist and are reachable from the 'Inputs'.")
    exit(1)

# Extract NPU part
# Inputs: Split node's input tensors + passthrough inputs (attention_mask)
# Outputs: Original model outputs
npu_input_names = sorted(list(set(split_input_tensor_names + passthrough_input_names)))  # Merge and deduplicate inputs
print(f"\nExtracting NPU part...")
print(f"  Inputs: {npu_input_names}")
print(f"  Outputs: {original_output_names}")
try:
    onnx.utils.extract_model(
        original_onnx_path,
        npu_model_path,
        input_names=npu_input_names,
        output_names=original_output_names,
        check_model=True  # Validate extracted model
    )
    print(f"NPU part saved to: {npu_model_path}")
except Exception as e:
    print(f"Error extracting or checking NPU part: {e}")
    print("Ensure all tensor names in 'Inputs' and 'Outputs' form a valid subgraph.")
    exit(1)

print("\nModel splitting process finished.")
