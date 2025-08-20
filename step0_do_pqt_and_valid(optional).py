import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchao.quantization.qat import FakeQuantizeConfig, IntXQuantizationAwareTrainingConfig
from torchao.quantization import quantize_

model_path = "ibm-granite/granite-embedding-278m-multilingual"

tokenizer = AutoTokenizer.from_pretrained(model_path)

original_model = AutoModel.from_pretrained(model_path)
original_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model.to(device)

quantized_model = AutoModel.from_pretrained(model_path)
quantized_model.to(device)

activation_config = FakeQuantizeConfig(
    torch.int8, "per_token", is_symmetric=False, scale_precision=torch.float32
)
weight_config = FakeQuantizeConfig(
    torch.int8, group_size=768, is_symmetric=True, scale_precision=torch.float32
)
qat_config = IntXQuantizationAwareTrainingConfig(activation_config, weight_config)

quantize_(quantized_model, qat_config)

ds = load_dataset("sentence-transformers/stackexchange-duplicates", "title-title-pair")

# Calculate the number of rows for the first 20%
total_rows = len(ds['train'])
subset_rows = int(total_rows * 0.2)
print('total lens: ', total_rows)

train_ds = ds['train']
train_ds.shuffle(seed=42)  # Shuffle the dataset for better training
train_ds.select(range(subset_rows))

sentences = train_ds['title1'] + train_ds['title2']

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

text_dataset = TextDataset(sentences)

batch_size = 40  # Adjust based on your hardware
dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(quantized_model.parameters(), lr=1e-5)  # Small learning rate for alignment

num_epochs = 1  # Adjust as needed; start with 1 and monitor loss

quantized_model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_output_orig = original_model(**inputs)
            orig_emb = model_output_orig[0][:, 0]
            orig_emb = F.normalize(orig_emb, dim=1)

        model_output_quant = quantized_model(**inputs)
        quant_emb = model_output_quant[0][:, 0]
        quant_emb = F.normalize(quant_emb, dim=1)

        cos_sim = F.cosine_similarity(orig_emb, quant_emb, dim=1)
        loss = (1 - cos_sim).sum() # .mean() if you like, makes not much different

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")


quantized_model.save_pretrained("int8_granite_embedding_278m_multilingual")
# After training, the quantized_model has been aligned. To deploy, you may need to convert it to a truly quantized model using same config.

# valid:
'''
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Model paths
original_model_path = "ibm-granite/granite-embedding-278m-multilingual"
quantized_model_path = "./int8_granite_embedding_278m_multilingual"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_path)

# Load models
original_model = AutoModel.from_pretrained(original_model_path)
quantized_model = AutoModel.from_pretrained(quantized_model_path)

# Set models to evaluation mode
original_model.eval()
quantized_model.eval()

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model.to(device)
quantized_model.to(device)

# Load dataset
ds = load_dataset("sentence-transformers/stackexchange-duplicates", "title-title-pair") # you may want to change to another dataset, as we are currently testing the behavior.

# Custom dataset to flatten the titles
class FlattenTitlesDataset(Dataset):
    def __init__(self, hf_dataset, split='train'):
        self.data = []
        for example in hf_dataset[split]:
            self.data.append(example['title1'])
            self.data.append(example['title2'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset
dataset = FlattenTitlesDataset(ds)

# Randomly select 10% of the dataset
np.random.seed(42)  # For reproducibility
indices = np.random.choice(len(dataset), size=int(0.01 * len(dataset)), replace=False)
sampler = SubsetRandomSampler(indices)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

# Compute cosine similarities
cos_similarities = []

with torch.no_grad():
    for batch in dataloader:
        # Tokenize batch
        tokenized_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        
        # Original model embeddings
        model_output_orig = original_model(**tokenized_inputs)
        emb_orig = model_output_orig[0][:, 0]  # CLS pooling
        emb_orig = torch.nn.functional.normalize(emb_orig, dim=1)
        
        # Quantized model embeddings
        model_output_quant = quantized_model(**tokenized_inputs)
        emb_quant = model_output_quant[0][:, 0]  # CLS pooling
        emb_quant = torch.nn.functional.normalize(emb_quant, dim=1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_orig, emb_quant, dim=1)
        cos_similarities.extend(cos_sim.cpu().numpy())

# Convert to numpy array
cos_similarities = np.array(cos_similarities)

# Calculate average cosine similarity
avg_cos_sim = np.mean(cos_similarities)
print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")

# Fit normal distribution
mu, sigma = stats.norm.fit(cos_similarities)

# Plot histogram and normal distribution fit
plt.figure(figsize=(10, 6))
sns.histplot(cos_similarities, kde=False, stat="density", bins=50, label="Cosine Similarity")
x = np.linspace(min(cos_similarities), max(cos_similarities), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal Fit (μ={mu:.4f}, σ={sigma:.4f})')
plt.title("Distribution of Cosine Similarities Between Original and Quantized Model")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("cosine_similarity_distribution.png")
plt.close()

print("Distribution plot saved as 'cosine_similarity_distribution.png'")
'''
