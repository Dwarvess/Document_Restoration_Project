import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pathlib

# --- YOL AYARI (Dosya bulamama sorununu çözer) ---
# Şu anki dosyanın olduğu yeri (src) sisteme tanıtıyoruz
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

# DÜZELTME BURADA: Dosya adımız 'gan_model.py' idi
from gan_model import GeneratorUNet 
# DÜZELTME BURADA: Dataset sınıfını doğrudan buradan çağıralım, hata riskini azaltır
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# --- AYARLAR ---
# Şampiyon modelin adını buraya yaz (20 mi 25 mi?)
MODEL_NAME = "generator_epoch_5.pth" 
SHOWCASE_COUNT = 3  # Kaç satır örnek olsun?
IMG_SIZE = 512

cuda = True if torch.cuda.is_available() else False

# --- DATASET SINIFI (create_showcase için özel) ---
class ShowcaseDataset(Dataset):
    def __init__(self, root, mode="test"):
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # Test klasör yolları
        self.files = sorted(os.listdir(os.path.join(root, mode, "source")))
        self.src_dir = os.path.join(root, mode, "source")
        self.tgt_dir = os.path.join(root, mode, "target")

    def __getitem__(self, index):
        img_name = self.files[index]
        img_A = Image.open(os.path.join(self.src_dir, img_name)).convert("RGB")
        img_B = Image.open(os.path.join(self.tgt_dir, img_name)).convert("RGB")
        
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return len(self.files)

def create_showcase():
    print("--- GÖRSEL KANIT OLUŞTURULUYOR ---")
    
    # Model Yükle
    generator = GeneratorUNet()
    if cuda: generator = generator.cuda()
    
    model_path = f"outputs/models/{MODEL_NAME}"
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if cuda else 'cpu')))
        print(f"✅ Model yüklendi: {MODEL_NAME}")
    else:
        print(f"HATA: Model dosyası bulunamadı -> {model_path}")
        return

    generator.eval()

    # Test Verisini Yükle
    # data/processed klasörünün varlığından emin ol
    data_path = "data/processed"
    if not os.path.exists(data_path):
        # Belki E: diskindedir, tam yol verelim
        data_path = os.path.join(os.path.dirname(CURRENT_DIR.parent), "data", "processed")

    dataloader = DataLoader(
        ShowcaseDataset(data_path, mode="test"),
        batch_size=SHOWCASE_COUNT, 
        shuffle=True, 
        num_workers=0, # Windows'ta hata vermemesi için 0 yapıyoruz
    )

    # Bir paket veri çek
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"HATA: Veri yüklenemedi. Klasör yollarını kontrol et. Detay: {e}")
        return

    real_A = Variable(batch["A"].type(torch.cuda.FloatTensor if cuda else torch.FloatTensor))
    real_B = Variable(batch["B"].type(torch.cuda.FloatTensor if cuda else torch.FloatTensor))
    
    # Model Tahmini Yap
    with torch.no_grad():
        fake_B = generator(real_A)

    # --- GÖRSELLEŞTİRME ---
    plt.figure(figsize=(12, 4 * SHOWCASE_COUNT))
    titles = ['Girdi (Bozuk)', 'Model Çıktısı (Bizimki)', 'Hedef (Gerçek)']

    for i in range(SHOWCASE_COUNT):
        # 1. Bozuk Resim
        plt.subplot(SHOWCASE_COUNT, 3, i*3 + 1)
        img_input = real_A[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        plt.imshow(img_input, cmap='gray')
        if i == 0: plt.title(titles[0], fontsize=14, fontweight='bold')
        plt.axis('off')

        # 2. Model Çıktısı
        plt.subplot(SHOWCASE_COUNT, 3, i*3 + 2)
        img_pred = fake_B[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        plt.imshow(img_pred, cmap='gray')
        if i == 0: plt.title(titles[1], fontsize=14, fontweight='bold', color='green')
        plt.axis('off')

        # 3. Gerçek (Hedef) Resim
        plt.subplot(SHOWCASE_COUNT, 3, i*3 + 3)
        img_target = real_B[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        plt.imshow(img_target, cmap='gray')
        if i == 0: plt.title(titles[2], fontsize=14, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/final_showcase.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Görsel Kanıt Kaydedildi: {save_path}")

if __name__ == "__main__":
    create_showcase()