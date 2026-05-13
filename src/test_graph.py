import os
import glob
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import sys

sys.path.append(r'C:\Yol\Bitirme\src') 
from gan_model import GeneratorUNet

# --- ÇİFT ÇİZGİ İÇİN 2 FARKLI KLASÖR YOLU ---
MODELS_DIR = r"C:\Users\orhan\Desktop\sonuclar_batch_8\models" # Kendi model klasörün
# 1. Eğitim (Train) Klasörü: Mavi Çizgi İçin
TRAIN_SRC_DIR = r"D:\Document_Restoration_Project\data\processed\train\source"
TRAIN_TGT_DIR = r"D:\Document_Restoration_Project\data\processed\train\target"
# 2. Test/Doğrulama (Validation) Klasörü: Turuncu Çizgi İçin (Yolu kontrol et!)
TEST_SRC_DIR = r"D:\Document_Restoration_Project\data\processed\test\source"
TEST_TGT_DIR = r"D:\Document_Restoration_Project\data\processed\test\target"

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_EDILECEK_FOTO_SAYISI = 20 # Her klasörden 20'şer tane seçecek

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def main():
    print(f"--- ÇİFT ÇİZGİLİ RÖNTGEN CİHAZI BAŞLATILDI: {DEVICE} ---")
    
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    model_files.sort(key=lambda x: int(x.split("epoch_")[-1].split(".")[0]))
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Train Havuzundan Rastgele Seçim
    all_train_images = [f for f in os.listdir(TRAIN_SRC_DIR) if f.endswith('.png')]
    train_images = random.sample(all_train_images, min(TEST_EDILECEK_FOTO_SAYISI, len(all_train_images)))
    
    # Test Havuzundan Rastgele Seçim
    all_test_images = [f for f in os.listdir(TEST_SRC_DIR) if f.endswith('.png')]
    test_images = random.sample(all_test_images, min(TEST_EDILECEK_FOTO_SAYISI, len(all_test_images)))
        
    print(f"✅ Train'den {len(train_images)}, Test'ten {len(test_images)} fotoğraf seçildi.")
    print(f"✅ {len(model_files)} kapsül çift taraflı taranıyor. Bu işlem biraz sürebilir...\n")

    generator = GeneratorUNet().to(DEVICE)
    generator.eval() 

    epoch_numbers = []
    train_psnr_scores = []
    test_psnr_scores = []

    with torch.no_grad():
        for model_path in model_files:
            epoch_num = int(model_path.split("epoch_")[-1].split(".")[0])
            
            checkpoint = torch.load(model_path, map_location=DEVICE)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            
            # 1. TRAIN PSNR HESAPLAMA (Mavi Çizgi)
            total_train_psnr = 0
            for img_name in train_images:
                real_A = Image.open(os.path.join(TRAIN_SRC_DIR, img_name)).convert("RGB")
                real_B = Image.open(os.path.join(TRAIN_TGT_DIR, img_name)).convert("RGB")
                real_A_tensor = transform(real_A).unsqueeze(0).to(DEVICE)
                real_B_tensor = transform(real_B).unsqueeze(0).to(DEVICE)
                fake_B_tensor = generator(real_A_tensor)
                total_train_psnr += calculate_psnr(fake_B_tensor, real_B_tensor).item()
            avg_train_psnr = total_train_psnr / len(train_images)
            
            # 2. TEST PSNR HESAPLAMA (Turuncu Çizgi)
            total_test_psnr = 0
            for img_name in test_images:
                real_A = Image.open(os.path.join(TEST_SRC_DIR, img_name)).convert("RGB")
                real_B = Image.open(os.path.join(TEST_TGT_DIR, img_name)).convert("RGB")
                real_A_tensor = transform(real_A).unsqueeze(0).to(DEVICE)
                real_B_tensor = transform(real_B).unsqueeze(0).to(DEVICE)
                fake_B_tensor = generator(real_A_tensor)
                total_test_psnr += calculate_psnr(fake_B_tensor, real_B_tensor).item()
            avg_test_psnr = total_test_psnr / len(test_images)

            # Değerleri listeye ekle
            epoch_numbers.append(epoch_num)
            train_psnr_scores.append(avg_train_psnr)
            test_psnr_scores.append(avg_test_psnr)
            
            print(f"-> Kapsül Epoch {epoch_num} | Train PSNR: {avg_train_psnr:.2f} dB | Test PSNR: {avg_test_psnr:.2f} dB")

    # --- BİREBİR İSTEDİĞİN ÇİFT ÇİZGİLİ GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, train_psnr_scores, marker='o', linestyle='--', color='#5A9BD5', linewidth=2, markersize=6, label='Eğitim Başarısı (Train)')
    plt.plot(epoch_numbers, test_psnr_scores, marker='s', linestyle='-', color='#ED7D31', linewidth=3, markersize=6, label='Gerçek Test Başarısı (Validation)')
    
    plt.title('Eğitim vs Test Performansı (Generalization Gap)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch Sayısı', fontsize=12)
    plt.ylabel('Kalite (PSNR - dB)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(epoch_numbers)
    plt.legend(loc='upper left', fontsize=11) # İsim etiketleri (Legend)
    plt.tight_layout()
    
    grafik_yolu = os.path.join(MODELS_DIR, "Generalization_Gap_Grafigi.png")
    plt.savefig(grafik_yolu, dpi=300)
    print(f"\n🎉 ÇİFT ÇİZGİLİ ZAFER! Grafik yüksek çözünürlükle kaydedildi: {grafik_yolu}")
    plt.show()

if __name__ == "__main__":
    main()