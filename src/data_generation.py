import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import random
import pathlib
import glob

# --- AYARLAR VE YOLLAR ---
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
BASE_DIR = CURRENT_FILE_PATH.parent.parent

RAW_TEXT_DIR = BASE_DIR / "data" / "raw_texts"
OUTPUT_BASE_DIR = BASE_DIR / "data" / "processed" / "train"
SOURCE_DIR = OUTPUT_BASE_DIR / "source"
TARGET_DIR = OUTPUT_BASE_DIR / "target"
RAG_CORPUS_DIR = BASE_DIR / "data" / "processed" / "rag_corpus"
FONTS_DIR = BASE_DIR / "data" / "fonts"

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(RAG_CORPUS_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)

IMG_SIZE = (512, 512) 
FONT_SIZE = 18
WORDS_PER_PAGE = 70 
BLACKLIST_FONTS = ["wingding", "webding", "symbol", "holomdl2", "marlett", "segmdl2", "mtsorts", "outlook", "bookshelf symbol", "eudc"]

# --- YARDIMCI FONKSİYONLAR ---
def is_font_safe(font_path):
    font_name = os.path.basename(font_path).lower()
    for bad_name in BLACKLIST_FONTS:
        if bad_name in font_name: return False
    return True

def get_random_font():
    all_fonts = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))
    if not all_fonts: return None
    for _ in range(10):
        choice = str(random.choice(all_fonts))
        if is_font_safe(choice): return choice
    return None

def safe_imwrite(path, img):
    try:
        is_success, im_buf_arr = cv2.imencode(".png", img)
        if is_success: 
            im_buf_arr.tofile(str(path))
            return True
        else: return False
    except Exception as e:
        print(f"HATA: {path} kaydedilemedi! {e}")
        return False

def clean_gutenberg_text(text):
    start_marker = "*** START OF THE PROJECT"
    end_marker = "*** END OF THE PROJECT"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1: text = text[start_idx + len(start_marker):]
    if end_idx != -1: text = text[:end_idx]
    return text.strip()

def create_clean_image(text, size=IMG_SIZE):
    img = Image.new('RGB', size, color=(255, 255, 255)) 
    draw = ImageDraw.Draw(img)
    selected_font_path = get_random_font()
    font = None
    if selected_font_path:
        try: font = ImageFont.truetype(selected_font_path, FONT_SIZE)
        except: font = None
    if font is None:
        try: font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except: font = ImageFont.load_default()

    margin = 40
    offset = 40
    lines = textwrap.wrap(text, width=40) 
    for line in lines:
        draw.text((margin, offset), line, font=font, fill=(0, 0, 0)) 
        offset += FONT_SIZE + 5
        if offset > size[1] - margin: break
    return np.array(img)

def get_text_coordinates(img):
    """Resimdeki YAZI piksellerinin yerini bulur."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y_indices, x_indices = np.where(gray < 200)
    if len(y_indices) > 0:
        return y_indices, x_indices
    return None, None

# --- GÜNCELLENMİŞ: MİKRO KİTAP KURDU SİMÜLASYONU ---
def simulate_micro_worm_tunnels(shape, start_point):
    """
    Çok kısa ve ince tüneller oluşturur (Hocanın isteği).
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = start_point
    
    # KALINLIK: Çok ince (1-harflik hasar için)
    thickness = random.randint(2, 4)
    
    # UZUNLUK: Çok kısa (Sadece bir ısırık gibi)
    max_steps = random.randint(15, 40) 
    
    for _ in range(max_steps):
        # Hafif kalınlık değişimi (doğallık için)
        current_thickness = max(1, thickness + random.randint(-1, 1))
        cv2.circle(mask, (cx, cy), current_thickness, 255, -1)
        
        # Adımlar daha küçük ve kıvrak
        dx = random.randint(-3, 3) 
        dy = random.randint(-3, 3)
        
        cx = np.clip(cx + dx, 0, w-1)
        cy = np.clip(cy + dy, 0, h-1)
        
        # %10 ihtimalle çok erken bırak (küçük nokta gibi kalsın)
        if random.random() > 0.90: break
            
    return mask

def apply_degradation(image):
    """HOCANIN İSTEĞİ: Sadece ince, lokalize kurt yenikleri."""
    noisy_img = image.copy()
    h, w, _ = noisy_img.shape

    # Yazıların yerini bul
    text_y, text_x = get_text_coordinates(image)
    has_text = text_y is not None

    # --- İPTAL EDİLENLER (Hoca İstemedi) ---
    # 1. Zemin Gürültüsü (Salt & Pepper) -> İPTAL
    # 2. Büyük Lekeler (Blobs) -> İPTAL

    # --- YENİ: MİKRO KİTAP KURTLARI ---
    # Sayfa başına çok sayıda ama küçük hasar
    num_worms = random.randint(30, 60) 
    
    if has_text:
        for _ in range(num_worms):
            # Rastgele bir yazı pikseli seç ve oradan başlat
            idx = random.randint(0, len(text_y)-1)
            start_point = (text_y[idx], text_x[idx])

            worm_mask = simulate_micro_worm_tunnels(noisy_img.shape, start_point)
            
            # RENK: BEYAZ (255,255,255) - Kağıt delinmiş gibi
            noisy_img[worm_mask == 255] = (255, 255, 255)

    # Hafif Blur (Hasarların kenarları çok keskin durmasın diye)
    if random.random() > 0.5:
        noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)

    return noisy_img

def process_book(filename):
    file_path = RAW_TEXT_DIR / filename
    book_id = filename.name.replace(".txt", "").replace(" ", "_").lower()
    print(f"--- {book_id} işleniyor ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f: full_text = f.read().replace('\n', ' ')
    except:
        with open(file_path, 'r', encoding='latin-1') as f: full_text = f.read().replace('\n', ' ')

    clean_text = clean_gutenberg_text(full_text)
    with open(RAG_CORPUS_DIR / f"clean_{filename.name}", 'w', encoding='utf-8') as f:
        f.write(clean_text)

    words = clean_text.split()
    count = 0
    MAX_PAGES_PER_BOOK = 2500 
    for i in range(0, len(words), WORDS_PER_PAGE):
        if count >= MAX_PAGES_PER_BOOK: break
        chunk = " ".join(words[i : i + WORDS_PER_PAGE])
        if len(chunk) < 50: continue 

        clean_img_pil = create_clean_image(chunk)
        noisy_img_cv = apply_degradation(clean_img_pil)
        img_name = f"{book_id}_{count:04d}.png"
        safe_imwrite(TARGET_DIR / img_name, cv2.cvtColor(clean_img_pil, cv2.COLOR_RGB2BGR))
        safe_imwrite(SOURCE_DIR / img_name, noisy_img_cv)
        count += 1
        if count % 200 == 0: print(f"   [GAN] {count} resim üretildi...")
    print(f"   TAMAMLANDI: {book_id} -> {count} resim.")

def main():
    print(f"Ana Dizin: {BASE_DIR}")
    fonts = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))
    if not fonts: print("UYARI: data/fonts klasörü boş! Varsayılan font kullanılacak.")
    else: print(f"Klasörde {len(fonts)} adet font bulundu.")
    txt_files = list(RAW_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print("HATA: Hiç .txt dosyası yok!")
        return
    print("Üretim başlıyor (HOCA ONAYLI MOD: Mikro Kurtlar)...")
    for txt_file in txt_files: process_book(txt_file)

if __name__ == "__main__":
    main()