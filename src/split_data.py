import os
import shutil
import random
import pathlib

# --- YOLLAR ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Kaynak (Şu an hepsi burada)
SOURCE_IMGS_DIR = PROCESSED_DIR / "train" / "source"
TARGET_IMGS_DIR = PROCESSED_DIR / "train" / "target"

# Hedef Klasörler
DIRS = ["train", "val", "test"]
SUBDIRS = ["source", "target"]

def split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Oran kontrolü
    if train_ratio + val_ratio + test_ratio != 1.0:
        print("HATA: Oranların toplamı 1.0 olmalı!")
        return

    print("Veri seti taranıyor...")
    # Tüm dosya isimlerini al (Sadece source'a bakmak yeterli, target'ta da aynısı var)
    filenames = [f for f in os.listdir(SOURCE_IMGS_DIR) if f.endswith('.png')]
    
    # Karıştır (Shuffle)
    random.shuffle(filenames)
    
    total_files = len(filenames)
    print(f"Toplam Resim Sayısı: {total_files}")
    
    # Adetleri hesapla
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count # Kalanlar test'e
    
    print(f"Planlanan Dağılım -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    # Geçici klasör oluşturma yerine, mevcut yapıyı koruyarak taşıma yapacağız.
    # Ancak önce val ve test klasörlerini oluşturmamız lazım.
    
    for d in DIRS:
        for s in SUBDIRS:
            os.makedirs(PROCESSED_DIR / d / s, exist_ok=True)
            
    # Dosyaları Taşı
    # Dikkat: Şu an hepsi 'train' içinde. O yüzden önce 'val' ve 'test' olanları oradan alıp taşıyacağız.
    # Kalanlar 'train'de kalacak.
    
    # Val Seti Taşıma
    val_files = filenames[train_count : train_count + val_count]
    move_files(val_files, "val")
    
    # Test Seti Taşıma
    test_files = filenames[train_count + val_count :]
    move_files(test_files, "test")
    
    print("\n✅ Veri ayırma işlemi tamamlandı!")
    print(f"Train klasöründe kalan: {len(os.listdir(PROCESSED_DIR / 'train' / 'source'))}")

def move_files(file_list, destination_split):
    print(f" -> {len(file_list)} dosya '{destination_split}' klasörüne taşınıyor...")
    for filename in file_list:
        # Source Taşı
        src_path = PROCESSED_DIR / "train" / "source" / filename
        dst_path = PROCESSED_DIR / destination_split / "source" / filename
        shutil.move(str(src_path), str(dst_path))
        
        # Target Taşı
        src_path = PROCESSED_DIR / "train" / "target" / filename
        dst_path = PROCESSED_DIR / destination_split / "target" / filename
        shutil.move(str(src_path), str(dst_path))

if __name__ == "__main__":
    split_dataset()