import os
import zipfile

def create_minimal_zip(zip_filename="seatbelt_minimal.zip"):
    # Define what to exclude to keep the size under 10MB
    exclude_dirs = {'.git', 'outputs', '__pycache__'}
    exclude_exts = {'.pt'} # Exclude heavy YOLO weights

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip excluded directories completely
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts):
                    continue
                if file == zip_filename or file == "zip_project.py":
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
                
    print(f"Created {zip_filename}")
    size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    create_minimal_zip()
