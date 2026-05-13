import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import os
import pathlib
import sys

# --- MODELİ ÇAĞIR ---
# gan_model.py dosyası src klasöründe olduğu için python path'e ekliyoruz
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from gan_model import GeneratorUNet

# --- AYARLAR ---
# DİKKAT: Buraya outputs/models klasöründeki EN SON dosyanın adını tam yazmalısın.
# Örneğin: "generator_epoch_29.pth" veya "generator_epoch_30.pth"
MODEL_NAME = "best_model_unet.pth"  # 

# Yollar
BASE_DIR = CURRENT_DIR.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / MODEL_NAME
INPUT_DIR = BASE_DIR / "inputs"          # Temizlenecek resimleri buraya at
OUTPUT_DIR = BASE_DIR / "outputs" / "inference_results"

# Klasörleri oluştur
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cihaz (GPU varsa kullanır, yoksa CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"--- Model Yükleniyor: {MODEL_NAME} ---")
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı!\nAranan yer: {MODEL_PATH}")
        print("Lütfen 'src/inference.py' içindeki MODEL_NAME satırını kontrol et.")
        return None

    # Modeli başlat
    generator = GeneratorUNet().to(DEVICE)
    
    # Ağırlıkları yükle
    # 1. Tüm eğitim paketini (checkpoint) güvenli bir şekilde belleğe al
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # 2. Paketin içinden sadece jeneratörün (generator) ağırlıklarını çekip modele yükle
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval() # Test moduna al (Önemli!)
    print("✅ Model başarıyla yüklendi ve hazır.")
    return generator

def process_images():
    generator = load_model()
    if generator is None: return

    # Resim dönüşümleri (Eğitimdeki 512x512 standardı)
    transform = transforms.Compose([
        transforms.Resize((512, 512), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Input klasöründeki resimleri bul
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"UYARI: '{INPUT_DIR}' klasörü boş!")
        print("Lütfen temizlenmesini istediğin bozuk resimleri 'inputs' klasörüne at.")
        return

    print(f"--- Toplam {len(image_files)} resim işlenecek ---")

    for img_name in image_files:
        img_path = INPUT_DIR / img_name
        
        # Resmi aç
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"HATA: {img_name} açılamadı. {e}")
            continue

        # Modele hazırla
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # --- SİHİRLİ AN (Tahmin) ---
        with torch.no_grad():
            fake_clean = generator(img_tensor)

        # Sonucu kaydet
        save_path = OUTPUT_DIR / f"restored_{img_name}"
        save_image(fake_clean, save_path, normalize=True)
        
        print(f" -> İşlendi: {img_name} >>> Kaydedildi.")

    print("\n✅ TÜM İŞLEMLER TAMAMLANDI!")
    print(f"Sonuçlar burada: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()