import os
import sys
from rknn.api import RKNN

# Initialize RKNN object
rknn = RKNN(verbose=True)

# Configure model for target platform
print('--> Configuring model')
rknn.config(target_platform='rk3588')
print('Done')

# Define input size list
input_size_list = [[1, 64]]

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(
    model='granite_automoel_cls_norm_b1_len64_npu.onnx',
    input_size_list=[[1, 64, 768], [1, 64, 768], [1, 64]],
    inputs=['/transformer/embeddings/Add_1_output_0', '/transformer/embeddings/position_embeddings/Gather_output_0', 'attention_mask']
)
if ret != 0:
    print('Failed to load model!')
    exit(ret)
print('Done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)  # Quantization disabled
if ret != 0:
    print('Failed to build model!')
    exit(ret)
print('Done')

# Export RKNN model
print('--> Exporting RKNN model')
ret = rknn.export_rknn("./ige_npu_part.rknn")
if ret != 0:
    print('Failed to export RKNN model!')
    exit(ret)
print('Done')
