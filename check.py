import pandas as pd

# 아무 camera_image parquet 파일 하나 불러서 컬럼 확인
df = pd.read_parquet(r"E:\dataset\testing\camera_image\11987368976578218644_1340_000_1360_000.parquet")

print(df.columns)
print(df.head())
