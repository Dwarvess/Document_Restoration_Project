import cv2
import numpy as np
import os
import random
import pathlib

# Yolları Ayarla
BASE_DIR = pathlib.Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data" / "processed" / "train" / "source"
TARGET_DIR = BASE_DIR / "data" / "processed" / "train" / "target"
OUTPUT_DIR = BASE_DIR / "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_showcase():
    # Klasördeki tüm resimleri listele
    all_files = os.listdir(SOURCE_DIR)
    
    # Rastgele 5 tane seç
    selected_files = random.sample(all_files, 5)
    
    rows = []
    
    print("Önizleme oluşturuluyor...")
    
    for filename in selected_files:
        # Bozuk ve Temiz hallerini oku
        src_img = cv2.imread(str(SOURCE_DIR / filename))
        tgt_img = cv2.imread(str(TARGET_DIR / filename))
        
        # Yan yana birleştir (Horizontal Stack)
        # Araya ince gri bir çizgi çekelim
        separator = np.full((src_img.shape[0], 10, 3), 128, dtype=np.uint8) # Gri ayraç
        combined = np.hstack((src_img, separator, tgt_img))
        
        rows.append(combined)
    
    # Tüm satırları alt alta birleştir (Vertical Stack)
    final_grid = np.vstack(rows)
    
    # Üstüne başlık eklemek istersen (Opsiyonel, basit olsun diye opencv ile yazmıyorum)
    # Kaydet
    save_path = OUTPUT_DIR / "dataset_preview.png"
    cv2.imwrite(str(save_path), final_grid)
    
    print(f"HARİKA! Vitrin görseli hazır: {save_path}")
    print("Bu resmi hocana göstermek için kullanabilirsin.")

if __name__ == "__main__":
    create_showcase()