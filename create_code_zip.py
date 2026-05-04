import zipfile
import os

print("Packaging code files... (this will take 2 seconds)")

with zipfile.ZipFile('kaggle_code_only.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Core Files
    for file in ['Main_DTFD_MIL.py', 'utils.py', 'requirements.txt', 'run_training.sh', 'report.tex']:
        if os.path.exists(file):
            zipf.write(file)
            
    # Core Directories (Excluding Dataset!)
    for d in ['Model', 'scratch']:
        if os.path.exists(d):
            for root, dirs, files in os.walk(d):
                for file in files:
                    file_path = os.path.join(root, file)
                    if '.gemini' in file_path or 'debug_log' in file_path:
                        continue
                    arcname = os.path.relpath(file_path, start='.')
                    zipf.write(file_path, arcname)

print("Successfully created kaggle_code_only.zip! It is very small and ready to upload.")
