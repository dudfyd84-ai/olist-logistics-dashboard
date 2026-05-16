import os
import zipfile
import shutil

def compress_csv_files(data_dir):
    backup_dir = os.path.join(data_dir, "..", "..", "Project1_data_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        csv_path = os.path.join(data_dir, file)
        zip_path = os.path.join(data_dir, file + ".zip")
        
        # 1. Zip 압축
        print(f"Compressing {file}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, arcname=file)
            
        # 2. 원본 백업 이동
        print(f"Moving {file} to backup...")
        shutil.move(csv_path, os.path.join(backup_dir, file))
        
    print(f"압축 및 원본 백업이 완료되었습니다. 백업 경로: {backup_dir}")

if __name__ == "__main__":
    data_dir = "c:/Users/테오/Documents/icb10/Project1/data"
    compress_csv_files(data_dir)
