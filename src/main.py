from dataframe import create_dataframe
from images import show_image, crop_image
import os

if __name__ == '__main__':
    leukemia_dir = "dataset/leukemia"  # 8491 images
    healthy_dir = "dataset/healthy"  # 3389 images

    df = create_dataframe(leukemia_dir, healthy_dir)
    print(df)  # [0;8490] are tumoral cells

    show_image(df, 1000)

    crop_image(df, 1000)