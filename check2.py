import os
import pandas as pd

camera_dir = r"E:\dataset\testing\camera_image"

camera_files = sorted([
    f for f in os.listdir(camera_dir)
    if f.endswith(".parquet")
])

camera_names = set()

print(f"🔍 {len(camera_files)}개의 .parquet 파일 중 카메라 이름 수집 중...")

for f_idx, filename in enumerate(camera_files):
    file_path = os.path.join(camera_dir, filename)
    try:
        df = pd.read_parquet(file_path)

        if 'key.camera_name' not in df.columns:
            print(f"⚠️ 파일 {filename} 에 'key.camera_name' 컬럼 없음!")
            continue

        # 유일한 camera_name 수집
        camera_names.update(df['key.camera_name'].unique())

    except Exception as e:
        print(f"⚠️ 파일 {filename} 읽기 실패: {e}")

print("\n🎥 발견된 camera_name 값들:")
for name in sorted(camera_names):
    print(f"- {name}")
