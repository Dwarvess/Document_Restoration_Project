import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gan_model import GeneratorUNet
import os
import sys

# --- OTOMATİK KLASÖR BULMA (Hata almamak için) ---
# Kodun olduğu klasörü bul
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Proje ana dizinini bul (src'den bir yukarı çık)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Inputs klasörünü ayarla
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "inputs", "deneme.png")
# Çıktı klasörünü ayarla
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LİSTE GÜNCELLENDİ ---
# Buraya 1-15'i de ekledik. Hepsini tek tek çizecek.
PAIRS = [
    (1, 5),
    (5, 10),
    (10, 15),
    (1, 15)  # <- Yeni eklenen genel karşılaştırma
]

def load_model(epoch, device):
    """Belirtilen epoch numarasina ait modeli yükler"""
    path = os.path.join(MODEL_DIR, f"generator_epoch_{epoch}.pth")
    if not os.path.exists(path):
        print(f"Warning: Model not found -> {path}")
        return None
    
    model = GeneratorUNet().to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_image(model, image_tensor):
    """Modeli kullanarak görüntüyü onarır"""
    with torch.no_grad():
        output = model(image_tensor)
    output_np = output.squeeze().cpu().numpy()
    if output_np.ndim == 3:
        output_np = np.transpose(output_np, (1, 2, 0))
    output_np = (output_np * 0.5 + 0.5) * 255
    return np.clip(output_np, 0, 255).astype(np.uint8)

def create_heatmap(img1, img2):
    """İki resim arasındaki farkı ısı haritasına çevirir"""
    diff = cv2.absdiff(img1, img2)
    # Farkı görünür kılmak için normalize et ve renklendir
    diff_enhanced = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    diff_enhanced = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    return diff_enhanced

def main():
    print(f"Device: {DEVICE}")
    print(f"Image Path: {TEST_IMAGE_PATH}")
    
    # 1. Orijinal Resmi Hazırla
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Image file not found at {TEST_IMAGE_PATH}")
        return
        
    original_img = cv2.imread(TEST_IMAGE_PATH)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(original_rgb).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Klasör yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. HER ÇİFT İÇİN AYRI DÖNGÜ VE KAYIT
    for epoch_a, epoch_b in PAIRS:
        print(f"Generating comparison for: Epoch {epoch_a} vs Epoch {epoch_b}...")
        
        # Modelleri yükle
        model_a = load_model(epoch_a, DEVICE)
        model_b = load_model(epoch_b, DEVICE)
        
        if model_a is None or model_b is None:
            continue

        # Çıktıları al
        out_a = process_image(model_a, img_tensor)
        out_b = process_image(model_b, img_tensor)
        
        # Gri ise RGB yap
        if out_a.ndim == 2: out_a = cv2.cvtColor(out_a, cv2.COLOR_GRAY2RGB)
        if out_b.ndim == 2: out_b = cv2.cvtColor(out_b, cv2.COLOR_GRAY2RGB)
        
        # Isı haritası
        heatmap = create_heatmap(out_a, out_b)
        
        # --- ÇİZİM (4 Sütunlu Tek Satır) ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # Sütun 1: Input
        axes[0].imshow(original_rgb)
        axes[0].set_title("Input (Damaged)", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Sütun 2: Epoch A
        axes[1].imshow(out_a)
        axes[1].set_title(f"Epoch {epoch_a} (Early Stage)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Sütun 3: Epoch B
        axes[2].imshow(out_b)
        axes[2].set_title(f"Epoch {epoch_b} (Later Stage)", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Sütun 4: Heatmap
        axes[3].imshow(heatmap)
        axes[3].set_title(f"Difference Heatmap\n(Epoch {epoch_a} -> {epoch_b})", fontsize=14, fontweight='bold', color='darkred')
        axes[3].axis('off')

        # Dosya ismi oluştur ve kaydet
        save_filename = f"compare_{epoch_a}_vs_{epoch_b}.png"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Hafızayı temizle
        
        print(f"✅ Saved: {save_filename}")

    print(f"\nAll images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()