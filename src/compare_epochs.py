import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gan_model import GeneratorUNet  # Doğru sınıf adı
import os

# ================= AYARLAR =================
# 1. Buraya test etmek istediğin hasarlı resmin tam adını yaz
# NOT: Bu dosyanın inputs klasöründe olduğundan emin ol.
TEST_IMAGE_PATH = "../inputs/deneme.png"

# 2. Karşılaştırılacak Modeller (Elinizdeki dosya isimlerine göre düzenleyin)
MODEL_PATHS = {
    "Epoch 1": "../outputs/models/generator_epoch_1.pth",
    "Epoch 3": "../outputs/models/generator_epoch_3.pth",
    "Epoch 5": "../outputs/models/generator_epoch_5.pth",
    "Epoch 10": "../outputs/models/generator_epoch_10.pth", 
    "Epoch 15": "../outputs/models/generator_epoch_15.pth"
}

OUTPUT_PATH = "../outputs/plots/epoch_comparison_poster.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def load_model(path, device):
    """Modeli yükler ve hazırlar"""
    if not os.path.exists(path):
        print(f"UYARI: Model bulunamadı -> {path}")
        return None
    
    # Modelin sınıfı GeneratorUNet olarak başlatılıyor
    model = GeneratorUNet().to(device)
    
    checkpoint = torch.load(path, map_location=device)
    # Eğer checkpoint içinde 'model_state_dict' anahtarı varsa onu yükle, yoksa direkt yükle
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_image(model, image_tensor):
    """Görüntüyü modele sokar ve sonucu alır"""
    with torch.no_grad():
        output = model(image_tensor)
    
    # Tensörden numpy array'e çevir (Görüntüleme için)
    output_np = output.squeeze().cpu().numpy()
    
    # Boyutları düzelt: (C, H, W) -> (H, W, C)
    if output_np.ndim == 3:
        output_np = np.transpose(output_np, (1, 2, 0))
    
    # Normalizasyonu geri al (-1, 1) -> (0, 255)
    output_np = (output_np * 0.5 + 0.5) * 255
    
    # Değerleri 0-255 arasına sıkıştır (clip) ve uint8 yap
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)
    return output_np

def main():
    print(f"İşlem cihazı: {DEVICE}")
    print(f"Hedef Resim: {TEST_IMAGE_PATH}")
    
    # 1. Girdi Resmini Hazırla
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"HATA: Resim bulunamadı! Lütfen yolu kontrol et: {TEST_IMAGE_PATH}")
        print("İPUCU: Terminalde 'src' klasörünün içinde olduğuna emin ol.")
        return

    # OpenCV ile oku (BGR formatında gelir)
    original_img = cv2.imread(TEST_IMAGE_PATH)
    if original_img is None:
        print("HATA: Resim okunamadı. Dosya bozuk olabilir.")
        return

    # BGR -> RGB Dönüşümü (Model renkli bekliyor)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # --- DÜZELTİLEN KISIM BAŞLANGICI ---
    # Görüntüyü PyTorch tensörüne çevir (3 Kanal RGB olarak)
    img_tensor = torch.from_numpy(original_img_rgb).float() / 255.0
    
    # Normalizasyon (-1 ile 1 arası yap)
    img_tensor = (img_tensor - 0.5) / 0.5
    
    # Boyutları ayarla: (H, W, C) -> (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # Batch boyutu ekle: (1, C, H, W) ve Cihaza gönder
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    # --- DÜZELTİLEN KISIM BİTİŞİ ---
    
    # Sonuçları saklayacak listeler
    results = []
    titles = []
    
    # İlk sıraya Orijinal Girdiyi ekle
    results.append(original_img_rgb)
    titles.append("Girdi (Hasarlı)")

    # 2. Modelleri Tek Tek Çalıştır
    for epoch_name, model_path in MODEL_PATHS.items():
        print(f"Model çalıştırılıyor: {epoch_name}...")
        model = load_model(model_path, DEVICE)
        
        if model:
            out_img = process_image(model, img_tensor)
            results.append(out_img)
            titles.append(epoch_name)
        else:
            # Model yoksa siyah boşluk koy
            results.append(np.zeros_like(original_img_rgb))
            titles.append(f"{epoch_name} (Yok)")

    # 3. Yan Yana Çizdir ve Kaydet
    print("Görsel oluşturuluyor...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 8)) # Genişlik, Yükseklik
    
    for i, ax in enumerate(axes):
        ax.imshow(results[i])
        ax.set_title(titles[i], fontsize=18, fontweight='bold')
        ax.axis('off') # Çerçeveyi kaldır

    plt.tight_layout()
    
    # Yüksek kalite kaydet (300 DPI)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ BİTTİ! Görsel şuraya kaydedildi: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()