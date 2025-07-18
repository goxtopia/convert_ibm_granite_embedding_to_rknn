import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
from transformers import AutoModel, AutoTokenizer
import os
from sklearn.metrics.pairwise import cosine_similarity # For similarity calculation

# --- Configuration ---
model_path = "ibm-granite/granite-embedding-278m-multilingual"
# Updated filename for this specific export
onnx_model_filename = "granite_automoel_cls_norm_b1_len64.onnx"
max_seq_length = 64 # Fixed sequence length, you could change len Here if you want.
fixed_batch_size = 1 # Fixed batch size

# --- 1. Define a Wrapper Module for Export ---
# This module encapsulates the transformer, CLS pooling, and normalization
class GraniteEmbeddingONNXWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # Load the base transformer model
        self.transformer = AutoModel.from_pretrained(model_path)
        self.transformer.eval() # Set to evaluation mode

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the transformer
        # Ensure return_dict=True for easy access to last_hidden_state
        model_output = self.transformer(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=True)
        # Extract the last hidden state
        last_hidden_state = model_output.last_hidden_state
        # Perform CLS pooling: select the embedding of the first token ([CLS])
        # Shape changes from [batch_size, seq_len, hidden_dim] to [batch_size, hidden_dim]
        cls_embedding = last_hidden_state[:, 0]
        # Normalize the embeddings along the hidden dimension
        normalized_embedding = F.normalize(cls_embedding, p=2, dim=1)
        # Return the final normalized embedding directly
        return normalized_embedding

# --- 2. Instantiate Wrapper, Load Tokenizer ---
print(f"Loading model and tokenizer: {model_path}...")
# Instantiate the wrapper module (this is what we'll export)
export_model = GraniteEmbeddingONNXWrapper(model_path)
export_model.to('cpu') # Ensure model is on CPU for export consistency
export_model.eval()

# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model wrapper and tokenizer loaded.")

# --- 3. Prepare Dummy Input for Fixed Shape Export ---
# Input must match the fixed shape: [batch_size, sequence_length] = [1, 64]
dummy_text = ["Sample text for ONNX export"] # Single sentence for batch size 1
print(f"Tokenizing dummy input with batch_size={fixed_batch_size}, max_length={max_seq_length}...")

# Tokenize with padding and truncation to the fixed length
dummy_tokenized = tokenizer(
    dummy_text,
    padding='max_length', # Pad to max_length
    truncation=True,      # Truncate to max_length
    max_length=max_seq_length,
    return_tensors='pt'   # Return PyTorch tensors
)

# Prepare the tuple of inputs required by the wrapper's forward method
# Must match the signature: forward(self, input_ids, attention_mask)
dummy_model_input_tuple = (dummy_tokenized['input_ids'], dummy_tokenized['attention_mask'])

# Verify shapes
assert dummy_model_input_tuple[0].shape == (fixed_batch_size, max_seq_length), f"Expected input_ids shape {(fixed_batch_size, max_seq_length)}, got {dummy_model_input_tuple[0].shape}"
assert dummy_model_input_tuple[1].shape == (fixed_batch_size, max_seq_length), f"Expected attention_mask shape {(fixed_batch_size, max_seq_length)}, got {dummy_model_input_tuple[1].shape}"

print("Dummy input prepared (fixed shape [1, 64]):")
print(f"- input_ids shape: {dummy_model_input_tuple[0].shape}")
print(f"- attention_mask shape: {dummy_model_input_tuple[1].shape}")

# --- 4. Export the Wrapper Model to ONNX (Fixed Shape) ---
# Define input/output names
input_names = ['input_ids', 'attention_mask']
# The wrapper returns one tensor directly, name it appropriately
output_names = ['sentence_embedding']

print(f"\nExporting model wrapper to ONNX (fixed shape): {onnx_model_filename}...")
try:
    torch.onnx.export(
        export_model,              # The nn.Module wrapper instance
        dummy_model_input_tuple,   # Tuple of inputs matching forward() signature
        onnx_model_filename,       # Output ONNX file path
        input_names=input_names,   # Input node names
        output_names=output_names, # Output node names
        # NO dynamic_axes specified - all dimensions are fixed
        opset_version=14,          # ONNX opset version (try lower if issues occur)
        export_params=True,        # Export trained weights
        do_constant_folding=True,  # Optional: optimize graph
    )
    print(f"✅ Model successfully exported to {onnx_model_filename} with fixed input shape [1, 64]")

    # --- 5. (Recommended) Verify the Exported ONNX Model ---
    print("\nVerifying the exported ONNX model...")

    # Load the ONNX model using ONNX Runtime
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(onnx_model_filename, sess_options, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    print("ONNX Runtime session created.")

    # Prepare test input (must be batch size 1, length 64)
    test_queries = [' Who made the song My achy breaky heart? '] # Single query
    print(f"Preparing test input: {test_queries[0]}")

    # Tokenize the single test query to the fixed shape [1, 64]
    test_tokenized_pt = tokenizer(
        test_queries,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )
    # Prepare ONNX input dictionary (names must match input_names)
    ort_inputs = {
        'input_ids': test_tokenized_pt['input_ids'].cpu().numpy(),
        'attention_mask': test_tokenized_pt['attention_mask'].cpu().numpy()
    }
    print("Test input shape for ONNX:", ort_inputs['input_ids'].shape) # Should be [1, 64]

    # --- Run inference with the original PyTorch code path ---
    print("Running inference with original PyTorch wrapper...")
    with torch.no_grad():
        # Pass through the wrapper module directly
        pytorch_embedding = export_model(test_tokenized_pt['input_ids'], test_tokenized_pt['attention_mask'])
        pytorch_embedding_np = pytorch_embedding.cpu().numpy()

    # --- Run inference with ONNX Runtime ---
    print("Running inference with ONNX Runtime...")
    # Pass the input dictionary, run expects output names list and input dict
    ort_outputs = ort_session.run(output_names, ort_inputs)
    # The result is a list, get the first (and only) output tensor
    onnx_embedding_np = ort_outputs[0]

    # --- Compare outputs ---
    print("\nComparing outputs:")
    print("PyTorch wrapper output shape:", pytorch_embedding_np.shape) # Should be [1, embedding_dim]
    print("ONNX output shape:", onnx_embedding_np.shape)     # Should be [1, embedding_dim]

    # Check if outputs are numerically close
    if np.allclose(pytorch_embedding_np, onnx_embedding_np, rtol=1e-4, atol=1e-5):
        print("✅ Verification successful: ONNX output matches PyTorch wrapper output within tolerance.")
    else:
        print("❌ Verification failed: Outputs differ significantly.")
        diff = np.abs(pytorch_embedding_np - onnx_embedding_np)
        print(f"   Max absolute difference: {np.max(diff)}")
        print(f"   Mean absolute difference: {np.mean(diff)}")

    # --- 6. Use the Fixed-Shape ONNX Model for Similarity ---
    print("\nCalculating similarity using the fixed-shape ONNX model (processing one by one)...")

    input_queries = [
        ' Who made the song My achy breaky heart? ',
        'summit define'
        ]
    input_passages = [
        "Achy Breaky Heart is a country song written by Don Von Tress. Originally titled Don't Tell My Heart and performed by The Marcy Brothers in 1991. ",
        "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain. : 2 the highest level. : 3 a meeting or series of meetings between the leaders of two or more governments."
        ]

    # Function to get embedding for a single text using the ONNX model
    def get_onnx_embedding(text):
        tokenized_np = tokenizer(
            [text], # Needs to be a list for batch dim 1
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='np' # Get numpy directly
        )
        onnx_input = {
            'input_ids': tokenized_np['input_ids'],
            'attention_mask': tokenized_np['attention_mask']
        }
        embedding = ort_session.run(output_names, onnx_input)[0]
        return embedding[0] # Return the embedding, removing the batch dim

    # Encode queries one by one
    print("Encoding queries individually...")
    query_embeddings_onnx = np.array([get_onnx_embedding(q) for q in input_queries])
    print("Query embeddings calculated (ONNX). Shape:", query_embeddings_onnx.shape)

    # Encode passages one by one
    print("Encoding passages individually...")
    passage_embeddings_onnx = np.array([get_onnx_embedding(p) for p in input_passages])
    print("Passage embeddings calculated (ONNX). Shape:", passage_embeddings_onnx.shape)

    # Calculate cosine similarity using sklearn
    cosine_scores_onnx = cosine_similarity(query_embeddings_onnx, passage_embeddings_onnx)
    print("\nCosine Similarity calculated using fixed-shape ONNX embeddings:")
    print(cosine_scores_onnx)

    # Compare with original PyTorch logic results (for sanity check)
    print("\nOriginal PyTorch logic similarity for comparison:")
    # Tokenize batches
    tokenized_queries_pt = tokenizer(input_queries, padding=True, truncation=True, return_tensors='pt')
    tokenized_passages_pt = tokenizer(input_passages, padding=True, truncation=True, return_tensors='pt')
    # Get embeddings using original logic
    with torch.no_grad():
        query_outputs_pt = export_model.transformer(**tokenized_queries_pt, return_dict=True)
        query_embeds_pt = F.normalize(query_outputs_pt.last_hidden_state[:, 0], p=2, dim=1)

        passage_outputs_pt = export_model.transformer(**tokenized_passages_pt, return_dict=True)
        passage_embeds_pt = F.normalize(passage_outputs_pt.last_hidden_state[:, 0], p=2, dim=1)

    # cosine_scores_pt = torch.functional.cos_sim(query_embeds_pt, passage_embeds_pt) # Use sentence-transformer util or sklearn
    # print(cosine_scores_pt.cpu().numpy())


except Exception as e:
    print(f"❌ An error occurred during ONNX export or usage: {e}")
    import traceback
    traceback.print_exc()
