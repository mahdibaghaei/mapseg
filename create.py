import os
import shutil

# مسیر دیتاست فعلی و مسیر هدف
dataset_root = "./dataset"  # مسیر فعلی دیتاست
mae_root = "./dataset_mae"  # مسیر جدید برای MAE

# ساخت پوشه مقصد
os.makedirs(mae_root, exist_ok=True)

# نگاشت پوشه‌ها به نام‌های مورد انتظار MAE
folder_map = {
    "ct_train": "src_1_train",
    "mr_train": "tgt_1_train",
    "ct_test": "src_1_test",
    "mr_test": "tgt_1_test"
}

for old_name, new_name in folder_map.items():
    old_path = os.path.join(dataset_root, old_name)
    new_path = os.path.join(mae_root, new_name)

    if os.path.exists(old_path):
        print(f"Copying {old_path} -> {new_path}")
        shutil.copytree(old_path, new_path)
    else:
        print(f"Warning: {old_path} does not exist!")

print("Dataset conversion complete!")
