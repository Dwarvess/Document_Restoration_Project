import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import pathlib
import datetime
import time
import math # PSNR hesabı için eklendi

# --- SENİN MODELİNİ ÇAĞIRIYORUZ ---
from gan_model import GeneratorUNet, Discriminator

# --- AYARLAR (Konfigürasyon) ---
EPOCHS = 20           # Toplam kaç tur dönecek?
BATCH_SIZE = 8        # Her seferde kaç resim işlesin?
LEARNING_RATE = 0.0002
IMG_SIZE = 512        
CHECKPOINT_INTERVAL = 1 

# Yollar
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
TRAIN_DIR_SRC = BASE_DIR / "data" / "processed" / "train" / "source"
TRAIN_DIR_TGT = BASE_DIR / "data" / "processed" / "train" / "target"
OUTPUT_MODEL_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_SAMPLE_DIR = BASE_DIR / "outputs" / "training_samples"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_SAMPLE_DIR, exist_ok=True)

# Cihaz Seçimi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- EĞİTİM CİHAZI: {DEVICE} ---")
if torch.cuda.is_available():
    print(f"Ekran Kartı: {torch.cuda.get_device_name(0)}")

# --- YARDIMCI: PSNR HESAPLAMA ---
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

# --- VERİ SETİ SINIFI ---
class ImageDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, transforms_=None):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.transform = transforms_
        self.files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])

    def __getitem__(self, index):
        img_name = self.files[index % len(self.files)]
        src_path = os.path.join(self.src_dir, img_name)
        tgt_path = os.path.join(self.tgt_dir, img_name)
        img_A = Image.open(src_path).convert("RGB")
        img_B = Image.open(tgt_path).convert("RGB")
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

# --- EĞİTİM FONKSİYONU ---
def train():
    # 1. Modelleri Başlat
    generator = GeneratorUNet().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Ağırlıkları başlatma
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 2. Kayıp Fonksiyonları
    criterion_GAN = nn.MSELoss() 
    criterion_pixel = nn.L1Loss() 

    # 3. Optimizasyoncular
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- ÖĞRENME HIZI PLANLAYICI (SCHEDULER) EKLENDİ ---
    # Her 10 epochta bir öğrenme hızını yarıya düşür (Fine-tuning için)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    # 4. Veri Yükleyici
    transforms_ = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])

    dataloader = DataLoader(
        ImageDataset(TRAIN_DIR_SRC, TRAIN_DIR_TGT, transforms_=transforms_),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Hata alırsan 0 yap
    )

    print(f"Toplam Eğitim Verisi: {len(dataloader)} batch")
    
    # --- DÖNGÜ BAŞLIYOR ---
    prev_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_psnr = 0
        batch_count = 0

        for i, batch in enumerate(dataloader):
            
            real_A = batch["A"].to(DEVICE) 
            real_B = batch["B"].to(DEVICE) 

            # Etiketler
            patch = (1, IMG_SIZE // 16, IMG_SIZE // 16)
            valid = torch.ones((real_A.size(0), *patch), requires_grad=False).to(DEVICE)
            fake = torch.zeros((real_A.size(0), *patch), requires_grad=False).to(DEVICE)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            fake_B = generator(real_A) 
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            loss_pixel = criterion_pixel(fake_B, real_B)

            # --- DİKKAT: L1 Loss Katsayısı 100 -> 1000 yapıldı ---
            loss_G = loss_GAN + (1000 * loss_pixel)

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --- PSNR Hesaplama ---
            with torch.no_grad():
                current_psnr = calculate_psnr(fake_B, real_B)
                epoch_psnr += current_psnr.item()
                batch_count += 1

            # Loglama
            batches_done = epoch * len(dataloader) + i
            batches_left = EPOCHS * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % 50 == 0:
                print(
                    f"\r[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                    f"[PSNR: {current_psnr.item():.2f} dB] ETA: {time_left}"
                )

        # Epoch bitince Scheduler çalıştır
        scheduler_G.step()
        scheduler_D.step()

        # Ortalama PSNR Yazdır
        avg_psnr = epoch_psnr / batch_count
        print(f"\n--> EPOCH {epoch} TAMAMLANDI. ORTALAMA PSNR: {avg_psnr:.2f} dB")

        # --- KAYDETME ---
        if epoch % CHECKPOINT_INTERVAL == 0:
            sample_img = torch.cat((real_A.data[0], fake_B.data[0], real_B.data[0]), -1)
            save_image(sample_img, OUTPUT_SAMPLE_DIR / f"epoch_{epoch}.png", normalize=True)
            torch.save(generator.state_dict(), OUTPUT_MODEL_DIR / f"generator_epoch_{epoch}.pth")
            print(f"-> Model kaydedildi: epoch_{epoch}")

if __name__ == "__main__":
    train()