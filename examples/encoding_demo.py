"""
Encoding Demonstration for WiMAE and ContraWiMAE

This notebook demonstrates how to use trained WiMAE and ContraWiMAE models
to encode wireless channel data and extract meaningful representations.

Convert this to a Jupyter notebook for interactive exploration.
"""

# %%
# Cell 1: Imports and Setup
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Add parent directory to path for imports
try:
    # For Python scripts
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    # For Jupyter notebooks
    sys.path.append(str(Path().cwd().parent))

# WiMAE imports
from wimae.encoding.encoder import WiMAEEncoder
from wimae.models.base import WiMAE
from wimae.models.contramae import ContraWiMAE
from wimae.training.data_utils import OptimizedPreloadedDataset, calculate_complex_statistics

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set style for prettier plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
# Cell 2: Data Preparation
data_path = "data/pretrain"
npz_files = list(Path(data_path).glob("*.npz"))

print(f"Available datasets: {len(npz_files)} cities")

# Load and examine a sample file
sample_file = npz_files[0]
with np.load(sample_file) as data:
    channels = data['channels']
    print(f"\nSample data from {sample_file.name}:")
    print(f"  • Shape: {channels.shape}")
    print(f"  • Data type: {channels.dtype}")
    print(f"  • Is complex: {np.iscomplexobj(channels)}")
    print(f"  • Value range: {np.abs(channels).min():.3f} to {np.abs(channels).max():.3f}")

# Convert to PyTorch tensor
sample_tensor = torch.from_numpy(channels[:100])  # Take first 100 samples
print(f"  • PyTorch tensor shape: {sample_tensor.shape}")

# %%
# Cell 3: Model Configuration and Loading
# Configuration matching training setup
config = {
    "model": {
        "input_dim": 1024,  # 32 * 32 = 1024 for flattened patches
        "encoder_dim": 64,
        "decoder_dim": 64,
        "contrastive_dim": 64,
        "num_heads": 8,
        "num_layers": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "mask_ratio": 0.75,
        "patch_size": [1, 16]
    },
    "data": {
        "channels": 1,
        "height": 32,
        "width": 32,
        "batch_size": 32
    }
}

# Create models (we'll load checkpoints if available)
wimae_model = WiMAE(
    input_dim=config["model"]["input_dim"],
    encoder_dim=config["model"]["encoder_dim"],
    decoder_dim=config["model"]["decoder_dim"],
    num_heads=config["model"]["num_heads"],
    num_layers=config["model"]["num_layers"],
    mlp_ratio=config["model"]["mlp_ratio"],
    dropout=config["model"]["dropout"],
    mask_ratio=config["model"]["mask_ratio"],
    patch_size=config["model"]["patch_size"]
)

contrawimae_model = ContraWiMAE(
    input_dim=config["model"]["input_dim"],
    encoder_dim=config["model"]["encoder_dim"],
    decoder_dim=config["model"]["decoder_dim"],
    contrastive_dim=config["model"]["contrastive_dim"],
    num_heads=config["model"]["num_heads"],
    num_layers=config["model"]["num_layers"],
    mlp_ratio=config["model"]["mlp_ratio"],
    dropout=config["model"]["dropout"],
    mask_ratio=config["model"]["mask_ratio"],
    patch_size=config["model"]["patch_size"],
    snr_min=0,
    snr_max=30
)

print("Models created successfully!")

# %%
# Cell 4: Checkpoint Loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Try to load checkpoints
wimae_loaded = False
contrawimae_loaded = False

# Look for WiMAE checkpoints
wimae_checkpoints = list(Path("checkpoints/wimae").glob("*.pth")) if Path("checkpoints/wimae").exists() else []
if wimae_checkpoints:
    try:
        # Load the most recent checkpoint
        latest_wimae = max(wimae_checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest_wimae, map_location=device)
        wimae_model.load_state_dict(checkpoint['model_state_dict'])
        wimae_model.to(device)
        wimae_model.eval()
        wimae_loaded = True
        print(f"Loaded WiMAE checkpoint: {latest_wimae.name}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    except Exception as e:
        print(f"Failed to load WiMAE checkpoint: {e}")

# Look for ContraWiMAE checkpoints
contrawimae_checkpoints = list(Path("checkpoints/contrawimae").glob("*.pth")) if Path("checkpoints/contrawimae").exists() else []
if contrawimae_checkpoints:
    try:
        # Load the most recent checkpoint
        latest_contrawimae = max(contrawimae_checkpoints, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest_contrawimae, map_location=device)
        contrawimae_model.load_state_dict(checkpoint['model_state_dict'])
        contrawimae_model.to(device)
        contrawimae_model.eval()
        contrawimae_loaded = True
        print(f"Loaded ContraWiMAE checkpoint: {latest_contrawimae.name}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    except Exception as e:
        print(f"Failed to load ContraWiMAE checkpoint: {e}")

if not (wimae_loaded or contrawimae_loaded):
    print("No checkpoints found. Using randomly initialized models for demonstration.")
    wimae_model.to(device)
    contrawimae_model.to(device)
    wimae_model.eval()
    contrawimae_model.eval()

# %%
# Cell 5: Data Preprocessing and Statistics
print("Preparing data for encoding...")

# Create dataset with statistics calculation
dataset = OptimizedPreloadedDataset(
    npz_files=npz_files[:3],  # Use first 3 cities for demo
    channels=config["data"]["channels"],
    height=config["data"]["height"],
    width=config["data"]["width"]
)

# Calculate statistics
stats = calculate_complex_statistics(dataset)
print(f"Calculated statistics:")
print(f"   Mean: {stats['mean']:.6f}")
print(f"   Std: {stats['std']:.6f}")

# Recreate dataset with normalization
dataset_normalized = OptimizedPreloadedDataset(
    npz_files=npz_files[:3],
    channels=config["data"]["channels"],
    height=config["data"]["height"],
    width=config["data"]["width"],
    statistics=stats
)

print(f"Dataset ready: {len(dataset_normalized)} samples")

# %%
# Cell 6: Encoding with WiMAE
print("Encoding samples with WiMAE...")

# Take a batch of samples for encoding
batch_size = 64
sample_indices = torch.randperm(len(dataset_normalized))[:batch_size]
samples = torch.stack([dataset_normalized[i] for i in sample_indices])
samples = samples.to(device)

print(f"Encoding batch shape: {samples.shape}")

with torch.no_grad():
    if wimae_loaded:
        # Use trained model
        wimae_output = wimae_model(samples, mask_ratio=0.0, return_reconstruction=False)
        wimae_encodings = wimae_output["encoded_features"]
    else:
        # Use random model for demo
        wimae_output = wimae_model(samples, mask_ratio=0.0, return_reconstruction=False)
        wimae_encodings = wimae_output["encoded_features"]

# Pool encodings (mean across sequence dimension)
wimae_embeddings = torch.mean(wimae_encodings, dim=1)  # (batch_size, encoder_dim)

print(f"WiMAE encodings shape: {wimae_encodings.shape}")
print(f"WiMAE embeddings shape: {wimae_embeddings.shape}")

# %%
# Cell 7: Encoding with ContraWiMAE
print("Encoding samples with ContraWiMAE...")

with torch.no_grad():
    if contrawimae_loaded:
        # Use trained model
        contrawimae_output = contrawimae_model(samples, mask_ratio=0.0, return_contrastive=True)
        contrawimae_encodings = contrawimae_output["encoded_features"]
        contrastive_features = contrawimae_output["contrastive_features"]
    else:
        # Use random model for demo
        contrawimae_output = contrawimae_model(samples, mask_ratio=0.0, return_contrastive=True)
        contrawimae_encodings = contrawimae_output["encoded_features"]
        contrastive_features = contrawimae_output["contrastive_features"]

# Pool encodings
contrawimae_embeddings = torch.mean(contrawimae_encodings, dim=1)  # (batch_size, encoder_dim)
contrastive_embeddings = torch.mean(contrastive_features, dim=1)   # (batch_size, contrastive_dim)

print(f"ContraWiMAE encodings shape: {contrawimae_encodings.shape}")
print(f"ContraWiMAE embeddings shape: {contrawimae_embeddings.shape}")
print(f"Contrastive embeddings shape: {contrastive_embeddings.shape}")

# %%
# Cell 8: Visualization with PCA
print("Visualizing embeddings with PCA...")

# Convert to numpy for sklearn
wimae_emb_np = wimae_embeddings.cpu().numpy()
contrawimae_emb_np = contrawimae_embeddings.cpu().numpy()
contrastive_emb_np = contrastive_embeddings.cpu().numpy()

# Apply PCA
pca_wimae = PCA(n_components=2)
pca_contrawimae = PCA(n_components=2)
pca_contrastive = PCA(n_components=2)

wimae_pca = pca_wimae.fit_transform(wimae_emb_np)
contrawimae_pca = pca_contrawimae.fit_transform(contrawimae_emb_np)
contrastive_pca = pca_contrastive.fit_transform(contrastive_emb_np)

# Create city labels for coloring
city_labels = []
sample_idx = 0
for i, npz_file in enumerate(npz_files[:3]):
    with np.load(npz_file) as data:
        n_samples = min(len(data['channels']), batch_size - sample_idx)
        city_labels.extend([f"City_{i}"] * n_samples)
        sample_idx += n_samples
        if sample_idx >= batch_size:
            break

# Plot PCA results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# WiMAE PCA
scatter1 = axes[0].scatter(wimae_pca[:, 0], wimae_pca[:, 1], c=range(len(city_labels)), 
                          cmap='tab10', alpha=0.7, s=50)
axes[0].set_title(f'WiMAE Embeddings (PCA)\nExplained Variance: {pca_wimae.explained_variance_ratio_.sum():.2%}')
axes[0].set_xlabel('First Principal Component')
axes[0].set_ylabel('Second Principal Component')
axes[0].grid(True, alpha=0.3)

# ContraWiMAE PCA
scatter2 = axes[1].scatter(contrawimae_pca[:, 0], contrawimae_pca[:, 1], c=range(len(city_labels)), 
                          cmap='tab10', alpha=0.7, s=50)
axes[1].set_title(f'ContraWiMAE Embeddings (PCA)\nExplained Variance: {pca_contrawimae.explained_variance_ratio_.sum():.2%}')
axes[1].set_xlabel('First Principal Component')
axes[1].set_ylabel('Second Principal Component')
axes[1].grid(True, alpha=0.3)

# Contrastive PCA
scatter3 = axes[2].scatter(contrastive_pca[:, 0], contrastive_pca[:, 1], c=range(len(city_labels)), 
                          cmap='tab10', alpha=0.7, s=50)
axes[2].set_title(f'Contrastive Features (PCA)\nExplained Variance: {pca_contrastive.explained_variance_ratio_.sum():.2%}')
axes[2].set_xlabel('First Principal Component')
axes[2].set_ylabel('Second Principal Component')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("PCA visualization completed!")

# %%
# Cell 9: Embedding Analysis and Statistics
print("Embedding Analysis")
print("=" * 40)

# Compute embedding statistics
def analyze_embeddings(embeddings, name):
    embeddings_np = embeddings.cpu().numpy()
    print(f"\n{name} Analysis:")
    print(f"   Shape: {embeddings_np.shape}")
    print(f"   Mean: {embeddings_np.mean():.6f}")
    print(f"   Std: {embeddings_np.std():.6f}")
    print(f"   Min: {embeddings_np.min():.6f}")
    print(f"   Max: {embeddings_np.max():.6f}")
    
    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings_np)
    avg_similarity = (sim_matrix.sum() - np.trace(sim_matrix)) / (sim_matrix.size - sim_matrix.shape[0])
    print(f"   Avg Cosine Similarity: {avg_similarity:.6f}")

analyze_embeddings(wimae_embeddings, "WiMAE Embeddings")
analyze_embeddings(contrawimae_embeddings, "ContraWiMAE Embeddings")
analyze_embeddings(contrastive_embeddings, "Contrastive Features")

# %%
# Cell 10: Reconstruction Quality Assessment
print("Assessing Reconstruction Quality...")

# Perform reconstruction with both models
with torch.no_grad():
    # WiMAE reconstruction
    wimae_recon_output = wimae_model(samples, mask_ratio=0.0, return_reconstruction=True)
    wimae_reconstructed = wimae_recon_output["reconstructed_patches"]
    
    # ContraWiMAE reconstruction
    contrawimae_recon_output = contrawimae_model(samples, mask_ratio=0.0, return_reconstruction=True)
    contrawimae_reconstructed = contrawimae_recon_output["reconstructed_patches"]

# Compute reconstruction errors
mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

# Get original patches for comparison
wimae_patches = wimae_model.patcher(samples)
contrawimae_patches = contrawimae_model.patcher(samples)

wimae_mse = mse_loss(wimae_reconstructed, wimae_patches)
wimae_mae = mae_loss(wimae_reconstructed, wimae_patches)

contrawimae_mse = mse_loss(contrawimae_reconstructed, contrawimae_patches)
contrawimae_mae = mae_loss(contrawimae_reconstructed, contrawimae_patches)

print(f"Reconstruction Quality:")
print(f"   WiMAE     - MSE: {wimae_mse:.6f}, MAE: {wimae_mae:.6f}")
print(f"   ContraWiMAE - MSE: {contrawimae_mse:.6f}, MAE: {contrawimae_mae:.6f}")

# %%
# Cell 11: Encoding API Demonstration
print("High-Level Encoding API Demonstration")
print("=" * 45)

# Create encoder instances (if checkpoints are available)
if wimae_loaded:
    encoder_config = {
        "model_path": str(latest_wimae),
        "data": config["data"],
        "device": str(device)
    }
    
    print("Example configuration for WiMAEEncoder:")
    print(f"   Model path: {encoder_config['model_path']}")
    print(f"   Device: {encoder_config['device']}")
    print(f"   Data config: {encoder_config['data']}")
    
    # Note: Actual encoder usage would be:
    # encoder = WiMAEEncoder(config=encoder_config)
    # encodings = encoder.encode_data(data_path="path/to/data.npz")

print("\nUsage Example:")
print("""
# Initialize encoder
encoder = WiMAEEncoder(config={
    'model_path': 'checkpoints/wimae/best_model.pth',
    'data': {'channels': 1, 'height': 32, 'width': 32},
    'device': 'cuda'
})

# Encode data
encodings = encoder.encode_data('data/pretrain/city_0_newyork_channels.npz')
""")

# %%
# Cell 12: Summary and Next Steps
print("Encoding Summary")
print("=" * 30)

print("Successfully demonstrated:")
print("• Loading trained WiMAE and ContraWiMAE models")
print("• Preprocessing wireless channel data")
print("• Extracting embeddings from both models")
print("• Visualizing embeddings with PCA")
print("• Analyzing embedding properties")
print("• Assessing reconstruction quality")

print(f"\nKey Results:")
if wimae_loaded and contrawimae_loaded:
    print("• Both models loaded from trained checkpoints")
    print(f"• WiMAE reconstruction MSE: {wimae_mse:.6f}")
    print(f"• ContraWiMAE reconstruction MSE: {contrawimae_mse:.6f}")
else:
    print("• Models used with random initialization (demo mode)")
    
print(f"• Embedding dimensions: {wimae_embeddings.shape[1]} (encoder), {contrastive_embeddings.shape[1]} (contrastive)")
print(f"• Processed {batch_size} samples from {len(npz_files[:3])} cities")

print("\nNext Steps:")
print("• Use embeddings for downstream tasks (beam prediction, LoS classification)")
print("• Experiment with different mask ratios during encoding")
print("• Compare embeddings across different SNR conditions")
print("• Analyze embedding clustering by geographical/environmental factors")
print("• Fine-tune models on task-specific data") 