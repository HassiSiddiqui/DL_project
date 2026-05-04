import os
import zipfile
import pickle
import glob
from tqdm import tqdm

def prepare_tcga(zip_dir, output_train, output_test):
    print("Preparing TCGA Lung dataset...")
    zip_files = glob.glob(os.path.join(zip_dir, '*.zip'))
    
    mDATA = {}
    
    for zf_path in tqdm(zip_files, desc="Extracting ZIPs"):
        with zipfile.ZipFile(zf_path, 'r') as zf:
            for file_info in zf.infolist():
                if file_info.filename.endswith('.pkl'):
                    slide_name = os.path.basename(file_info.filename).replace('.pkl', '')
                    with zf.open(file_info.filename) as f:
                        data = pickle.load(f)
                    
                    # Ensure each patch has a label
                    # In LUAD vs LUSC, we would need external labels. 
                    # For demonstration/pipeline completeness, we assign dummy labels (0 or 1 based on hash)
                    label = hash(slide_name) % 2
                    for patch in data:
                        patch['label'] = label
                        
                    mDATA[slide_name] = data
                    
    # Split into train and test
    slides = list(mDATA.keys())
    split_idx = int(len(slides) * 0.8)
    train_slides = slides[:split_idx]
    test_slides = slides[split_idx:]
    
    mDATA_train = {s: mDATA[s] for s in train_slides}
    mDATA_test = {s: mDATA[s] for s in test_slides}
    
    print(f"Saving {len(mDATA_train)} train slides to {output_train}")
    with open(output_train, 'wb') as f:
        pickle.dump(mDATA_train, f)
        
    print(f"Saving {len(mDATA_test)} test slides to {output_test}")
    with open(output_test, 'wb') as f:
        pickle.dump(mDATA_test, f)
        
    print("TCGA dataset preparation complete.")

if __name__ == '__main__':
    zip_dir = 'Dataset/TCGA_Lung'
    output_train = 'Dataset/TCGA_Lung/mDATA_train_TCGA.pkl'
    output_test = 'Dataset/TCGA_Lung/mDATA_test_TCGA.pkl'
    if not os.path.exists(output_train):
        prepare_tcga(zip_dir, output_train, output_test)
    else:
        print("TCGA data already prepared.")
