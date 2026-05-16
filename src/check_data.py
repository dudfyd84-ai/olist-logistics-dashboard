"""
이 파일은 프로젝트 데이터의 무결성을 확인하기 위한 스크립트입니다.
data 폴더 내의 모든 CSV 파일을 로드하여 데이터의 구조, 샘플 행, 요약 통계 및 중복 데이터를 출력합니다.
"""

import pandas as pd
import os
import glob
from tabulate import tabulate

def check_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    print(f"Total CSV files found: {len(csv_files)}")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"Checking file: {file_name}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(file_path)
            
            print(f"\n1. Shape: {df.shape}")
            
            print("\n2. Top 5 rows:")
            print(tabulate(df.head(), headers='keys', tablefmt='psql'))
            
            print("\n3. Bottom 5 rows:")
            print(tabulate(df.tail(), headers='keys', tablefmt='psql'))
            
            print("\n4. Info:")
            df.info()
            
            print(f"\n5. Duplicated rows: {df.duplicated().sum()}")
            
            print("\n6. Descriptive Statistics (Numerical):")
            if not df.select_dtypes(include=['number']).empty:
                print(tabulate(df.describe(), headers='keys', tablefmt='psql'))
            else:
                print("No numerical columns found.")
                
            print("\n7. Descriptive Statistics (Categorical):")
            if not df.select_dtypes(include=['object']).empty:
                print(tabulate(df.describe(include=['object']), headers='keys', tablefmt='psql'))
            else:
                print("No categorical columns found.")
                
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    check_data()
