import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import pathlib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys
import glob
import re

# --- YOLLAR ---
# Bu dosyanın (plot_progress.py) olduğu yer: src klasörü
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
# Projenin ana klasörü (src'nin bir üstü)
BASE_DIR = CURRENT_DIR.parent

MODELS_DIR = BASE_DIR / "outputs" / "models"
TEST_DIR_SRC = BASE_DIR / "data" / "processed" / "test" / "source"
TEST_DIR_TGT = BASE_DIR / "data" / "processed" / "test" / "target"
OUTPUT_PLOT_DIR = BASE_DIR / "outputs" / "plots"

# LOG DOSYASI BURADA ARANACAK: Ana klasördeki my_logs.txt
LOG_FILE_PATH = BASE_DIR / "my_logs.txt"

# Model dosyasını import et
sys.path.append(str(CURRENT_DIR))
from gan_model import GeneratorUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

# --- DATASET SINIFI ---
class QuickTestDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, limit=20): 
        # Test için 20 resim yeterli (Hızlı olsun diye)
        self.files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])[:limit]
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        img_name = self.files[index]
        img_A = self.transform(Image.open(os.path.join(self.src_dir, img_name)).convert("RGB"))
        img_B = self.transform(Image.open(os.path.join(self.tgt_dir, img_name)).convert("RGB"))
        return img_A, img_B

    def __len__(self):
        return len(self.files)

def parse_training_logs():
    """Txt dosyasından Eğitim PSNR değerlerini okur."""
    train_scores = {}
    
    print(f"Log dosyası aranıyor: {LOG_FILE_PATH}")
    
    if not os.path.exists(LOG_FILE_PATH):
        print(f"UYARI: 'my_logs.txt' bulunamadı! Sadece test grafiği çizilecek.")
        print("İPUCU: my_logs.txt dosyasını proje ana klasörüne taşıdın mı?")
        return train_scores

    with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        # Regex ile "EPOCH X ... PSNR: Y dB" satırlarını bul
        matches = re.findall(r'EPOCH (\d+) TAMAMLANDI.*?PSNR: ([\d.]+) dB', content, re.DOTALL)
        
        for ep, score in matches:
            train_scores[int(ep)] = float(score)
            
    print(f"-> Log dosyasından {len(train_scores)} adet epoch verisi okundu.")
    return train_scores

def calculate_test_scores():
    """Her model dosyası için gerçek Test PSNR değerini hesaplar."""
    test_scores = {}
    # Sadece epoch numarası içeren dosyaları bul
    model_files = glob.glob(str(MODELS_DIR / "generator_epoch_*.pth"))
    
    if not model_files:
        print("HATA: Hiç .pth modeli bulunamadı!")
        return {}

    test_loader = DataLoader(
        QuickTestDataset(TEST_DIR_SRC, TEST_DIR_TGT, limit=20), 
        batch_size=4, shuffle=False
    )
    
    generator = GeneratorUNet().to(DEVICE)
    
    print(f"-> {len(model_files)} adet model dosyası test ediliyor...")
    
    for f in sorted(model_files):
        match = re.search(r'epoch_(\d+)', f)
        if match:
            epoch = int(match.group(1))
            
            # Modeli yükle
            try:
                generator.load_state_dict(torch.load(f, map_location=DEVICE))
                generator.eval()
                
                batch_psnrs = []
                with torch.no_grad():
                    for real_A, real_B in test_loader:
                        real_A = real_A.to(DEVICE)
                        real_B = real_B.cpu().numpy() * 0.5 + 0.5
                        
                        fake_B = generator(real_A)
                        fake_B = fake_B.cpu().numpy() * 0.5 + 0.5
                        
                        for i in range(fake_B.shape[0]):
                            val = psnr(real_B[i].transpose(1, 2, 0), fake_B[i].transpose(1, 2, 0), data_range=1.0)
                            batch_psnrs.append(val)
                
                avg_psnr = np.mean(batch_psnrs)
                test_scores[epoch] = avg_psnr
                print(f"   Epoch {epoch}: Test PSNR = {avg_psnr:.2f} dB")
            except Exception as e:
                print(f"   Epoch {epoch} yüklenirken hata: {e}")
            
    return test_scores

def main():
    print("--- GRAFİK OLUŞTURUCU BAŞLATILDI ---")
    
    # 1. Verileri Topla
    train_data = parse_training_logs()
    test_data = calculate_test_scores()
    
    # Ortak Epochları Bul
    all_epochs = sorted(list(set(train_data.keys()) | set(test_data.keys())))
    
    if not all_epochs:
        print("HATA: Hiç veri bulunamadı! Ne log var ne model.")
        return

    epochs_x = []
    train_y = []
    test_y = []
    
    for ep in all_epochs:
        epochs_x.append(ep)
        train_y.append(train_data.get(ep, None)) # Veri yoksa boş bırak
        test_y.append(test_data.get(ep, None))

    # 2. Çizim Yap
    plt.figure(figsize=(10, 6))
    
    # Train Çizgisi (Mavi)
    # Eğer veri varsa çiz
    if any(train_y):
        clean_x = [x for x, y in zip(epochs_x, train_y) if y is not None]
        clean_y = [y for y in train_y if y is not None]
        plt.plot(clean_x, clean_y, marker='o', linestyle='--', color='#1f77b4', label='Eğitim Başarısı (Train)', alpha=0.7)
    
    # Test Çizgisi (Turuncu - Kalın)
    if any(test_y):
        clean_x = [x for x, y in zip(epochs_x, test_y) if y is not None]
        clean_y = [y for y in test_y if y is not None]
        plt.plot(clean_x, clean_y, marker='s', linestyle='-', color='#ff7f0e', linewidth=2.5, label='Gerçek Test Başarısı (Validation)')
    
    plt.title("Eğitim vs Test Performansı (Generalization Gap)", fontsize=14)
    plt.xlabel("Epoch Sayısı", fontsize=12)
    plt.ylabel("Kalite (PSNR - dB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    # 3. Kaydet
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    save_path = OUTPUT_PLOT_DIR / "final_comparison_graph.png"
    plt.savefig(save_path, dpi=300)
    
    print(f"\n✅ GRAFİK HAZIR: {save_path}")
    print("Grafiği açıp bakabilirsin!")

if __name__ == "__main__":
    main()