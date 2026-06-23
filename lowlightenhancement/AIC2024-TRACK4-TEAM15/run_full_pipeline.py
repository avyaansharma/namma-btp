import os
import shutil
import subprocess
from PIL import Image

def run_cmd(cmd, cwd=None):
    print(f"\n>>> Running: {cmd} in {cwd or 'root'}")
    env = os.environ.copy()
    if cwd:
        env["PYTHONPATH"] = os.path.abspath(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    # Run command and show stdout/stderr in real-time
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")

def clean_dir(path):
    if os.path.exists(path):
        print(f"Cleaning directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def main():
    root_dir = os.getcwd()
    venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")
    
    # Paths
    gen_img_path = r"C:\Users\Acer\OneDrive\Desktop\lowlightenhancement\image.png"
    input_dir = os.path.join(root_dir, "sample_dataset", "CVPR_test")
    input_img_path = os.path.join(input_dir, "test_image_N_001.png")
    
    nafnet_out_dir = os.path.join(root_dir, "sample_dataset", "NAFNet_Output", "cvpr_test")
    gsad_out_dir = os.path.join(root_dir, "sample_dataset", "GSAD_Output", "cvpr_test")
    dat_out_img = os.path.join(root_dir, "src", "lib", "infer_DAT", "results", "test_single_x4", "visualization", "Single", "test_image_N_001_x4.png")
    
    # 0. Preparation
    print("Step 0: Copying and converting generated image...")
    clean_dir(input_dir)
    # Open and convert to RGB
    img = Image.open(gen_img_path).convert("RGB")
    img.save(input_img_path)
    print(f"Saved input image to: {input_img_path}")
    
    # Clean output directories
    clean_dir(nafnet_out_dir)
    clean_dir(gsad_out_dir)
    clean_dir(os.path.join(root_dir, "pipeline_step_outputs"))
    
    # Clean DAT output dir if exists
    dat_vis_dir = os.path.dirname(dat_out_img)
    if os.path.exists(dat_vis_dir):
        shutil.rmtree(os.path.dirname(os.path.dirname(dat_vis_dir))) # clean results/test_single_x4
        
    # 1. Run NAFNet (Denoising & Deblurring)
    print("\n--- Step 1: Running NAFNet Inference ---")
    run_cmd(f'"{venv_python}" src/lib/infer_NAFNet/nafnet_inference.py', cwd=root_dir)
    
    nafnet_result = os.path.join(nafnet_out_dir, "test_image_N_001.png")
    if not os.path.exists(nafnet_result):
        raise FileNotFoundError(f"NAFNet output not found at: {nafnet_result}")
    print("NAFNet Inference Completed Successfully!")
    
    # 2. Run GSAD (Night-to-Day Enhancement)
    print("\n--- Step 2: Running GSAD Night-to-Day ---")
    gsad_cwd = os.path.join(root_dir, "src", "lib", "infer_GSAD")
    run_cmd(f'"{venv_python}" test_unpaired.py --input ../../../sample_dataset/NAFNet_Output/cvpr_test/ --save_dir ../../../sample_dataset/GSAD_Output/cvpr_test/', cwd=gsad_cwd)
    
    gsad_result = os.path.join(gsad_out_dir, "test_image_N_001.png")
    if not os.path.exists(gsad_result):
        raise FileNotFoundError(f"GSAD output not found at: {gsad_result}")
    print("GSAD Night-to-Day Completed Successfully!")
    
    # 3. Run DAT (4x Super Resolution)
    print("\n--- Step 3: Running DAT 4x Super Resolution ---")
    dat_cwd = os.path.join(root_dir, "src", "lib", "infer_DAT")
    run_cmd(f'"{venv_python}" basicsr/test.py -opt options/Test/convert_CVPR_test.yaml', cwd=dat_cwd)
    
    if not os.path.exists(dat_out_img):
        raise FileNotFoundError(f"DAT output not found at: {dat_out_img}")
    print("DAT Super Resolution Completed Successfully!")
    
    # 4. Copy step outputs to pipeline_step_outputs
    print("\n--- Step 4: Organizing Pipeline Stage Outputs ---")
    shutil.copy(input_img_path, os.path.join(root_dir, "pipeline_step_outputs", "1_original_input.png"))
    shutil.copy(nafnet_result, os.path.join(root_dir, "pipeline_step_outputs", "2_nafnet_enhanced.png"))
    shutil.copy(gsad_result, os.path.join(root_dir, "pipeline_step_outputs", "3_gsad_daylight.png"))
    shutil.copy(dat_out_img, os.path.join(root_dir, "pipeline_step_outputs", "4_dat_super_resolution.png"))
    print("Organized pipeline stage images in: pipeline_step_outputs/")
    print("\n=======================================================")
    print("All Enhancement Pipeline Steps Executed Successfully!")
    print("Check pipeline_step_outputs/ for all step visualizations.")
    print("=======================================================")

if __name__ == "__main__":
    main()
