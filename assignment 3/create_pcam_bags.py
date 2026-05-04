import os
import numpy as np
import pickle
import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, SamModel, SamProcessor
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

def main():
    print("=== Assignment 3: PatchCamelyon SAM+PLIP Builder ===")
    
    # 1. Download/Load Dataset (Subset of 10,000 images)
    print("Loading PatchCamelyon Dataset (Downloading if necessary)...")
    dataset = load_dataset("1aurent/PatchCamelyon", split="train")
    
    # Shuffle and select 1,000 patches (1/10th the size for massive speedup)
    dataset = dataset.shuffle(seed=42).select(range(1000))
    print(f"Loaded {len(dataset)} patches.")

    # 2. Group into MIL Bags (20 patches per bag -> 50 bags)
    num_bags = 50
    patches_per_bag = 20
    bags = []
    for i in range(num_bags):
        bag_patches = dataset[i*patches_per_bag : (i+1)*patches_per_bag]
        # A bag is tumor (1) if at least one patch is tumor (1)
        bag_label = 1 if 1 in bag_patches['label'] else 0
        bags.append({
            'images': bag_patches['image'],
            'label': bag_label,
            'name': f'PCAM_Bag_{i}'
        })
    print(f"Constructed {len(bags)} MIL Bags.")

    # 3. Load Foundation Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Models onto {device}...")
    
    # PLIP Model (Generates 512-dim features)
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    plip_model = CLIPModel.from_pretrained("vinid/plip").to(device)
    plip_model.eval()

    # SAM Model (We use the vision encoder to get semantic spatial features)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_model.eval()

    mDATA_PCAM = {}
    
    print("Extracting Features and Semantic Clusters...")
    for bag in tqdm(bags, desc="Processing Bags"):
        slide_data = []
        sam_embeddings = []
        plip_features = []
        
        # Batch process patches in the bag (100 patches)
        images = [img.convert("RGB") for img in bag['images']]
        
        with torch.no_grad():
            batch_size = 1
            
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                
                # Get PLIP Features
                plip_inputs = plip_processor(images=batch_imgs, return_tensors="pt").to(device)
                img_features = plip_model.get_image_features(**plip_inputs)
                
                if hasattr(img_features, 'image_embeds'):
                    batch_plip = img_features.image_embeds.cpu()
                elif hasattr(img_features, 'pooler_output'):
                    batch_plip = img_features.pooler_output.cpu()
                elif isinstance(img_features, tuple):
                    batch_plip = img_features[0].cpu()
                else:
                    batch_plip = img_features.cpu()
                plip_features.append(batch_plip)
                
                # Get SAM Embeddings
                sam_inputs = sam_processor(images=batch_imgs, return_tensors="pt").to(device)
                sam_outputs = sam_model.vision_encoder(sam_inputs.pixel_values)
                batch_sam = sam_outputs.last_hidden_state.mean(dim=[2, 3]).cpu().numpy()
                sam_embeddings.append(batch_sam)
                
            plip_features = torch.cat(plip_features, dim=0) # Shape: (100, 512)
            sam_embeddings = np.concatenate(sam_embeddings, axis=0) # Shape: (100, 256)
            
        # 4. K-Means Clustering on SAM Embeddings to form "Semantic Pseudo-bags"
        # DTFD-MIL uses Num_Dist pseudo-bags (e.g., 5 or 10)
        num_pseudo_bags = 5 
        kmeans = KMeans(n_clusters=num_pseudo_bags, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(sam_embeddings)
        
        # Save to list of dictionaries
        for i in range(len(images)):
            patch_dict = {
                'feature': plip_features[i],   # 512-dim tensor
                'cluster_id': cluster_labels[i], # SAM Semantic Cluster
                'label': bag['label'],
                'name': f"{bag['name']}_patch_{i}.jpg"
            }
            slide_data.append(patch_dict)
            
        mDATA_PCAM[bag['name']] = slide_data

    # Split into 80/20 Train/Test
    slides = list(mDATA_PCAM.keys())
    split_idx = int(len(slides) * 0.8)
    
    mDATA_train = {s: mDATA_PCAM[s] for s in slides[:split_idx]}
    mDATA_test = {s: mDATA_PCAM[s] for s in slides[split_idx:]}
    
    os.makedirs("assignment 3/Dataset", exist_ok=True)
    with open("assignment 3/Dataset/mDATA_train_PCAM.pkl", 'wb') as f:
        pickle.dump(mDATA_train, f)
    with open("assignment 3/Dataset/mDATA_test_PCAM.pkl", 'wb') as f:
        pickle.dump(mDATA_test, f)
        
    print("Done! PCAM dataset successfully converted for Assignment 3.")

if __name__ == "__main__":
    main()
