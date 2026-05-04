import zipfile
import os
from tqdm import tqdm

print("Calculating total file size. This will just take a moment...")
total_size = 0
files_to_zip = []

# Core Files
for file in ['Main_DTFD_MIL.py', 'utils.py', 'requirements.txt', 'run_training.sh', 'report.tex']:
    if os.path.exists(file):
        files_to_zip.append(file)
        total_size += os.path.getsize(file)

# Core Directories
for d in ['Model', 'scratch', 'Dataset']:
    if os.path.exists(d):
        for root, dirs, files in os.walk(d):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip the zip file itself if it's in the directory, and avoid temp folders
                if 'kaggle_dtfd_mil.zip' in file_path or '.gemini' in file_path or 'debug_log' in file_path:
                    continue
                files_to_zip.append(file_path)
                total_size += os.path.getsize(file_path)

print(f"Total size to zip: {total_size / (1024**3):.2f} GB")

# allowZip64=True is necessary for files/archives > 4GB
with zipfile.ZipFile('kaggle_dtfd_mil.zip', 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Zipping") as pbar:
        for file_path in files_to_zip:
            arcname = os.path.relpath(file_path, start='.')
            zipf.write(file_path, arcname)
            pbar.update(os.path.getsize(file_path))

print("Successfully created kaggle_dtfd_mil.zip!")
