import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. OCR (EASYOCR) FONKSİYONU ---
def run_ocr(image_path):
    print("--- 2. AŞAMA: OCR Motoru Yükleniyor ---")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)
    
    if not os.path.exists(image_path):
        return f"HATA: {image_path} konumunda görsel bulunamadı."
    
    print(f"📷 Görsel taranıyor: {os.path.basename(image_path)}...")
    result = reader.readtext(image_path, detail=0, paragraph=True)
    extracted_text = " ".join(result)
    return extracted_text


if __name__ == "__main__":
    print("--- OCR TEST TERMİNALİ ---")
    
    # 1. Okunacak GAN Çıktısı (Bir önceki adımda sabitlediğimiz yol)
    test_image_path = r"D:\Document_Restoration_Project\outputs\inference_results\restored_deneme.png"
    
    if test_image_path.strip():
        # OCR fonksiyonunu çağırıp metni al
        text = run_ocr(test_image_path)
        
        # Terminale yazdır
        print("\n📝 [OCR Çıktısı - Bu metin RAG'a gidecek]:")
        print("\033[93m" + text + "\033[0m")
        
        # --- YENİ: METNİ DOSYAYA KAYDETME KISMI ---
        # Kaydedilecek hedefin yolu
        output_txt_path = r"D:\Document_Restoration_Project\outputs\ocr_result.txt"
        
        # Dosyayı yazma ('w' - write) modunda aç ve metni kaydet. 
        # encoding="utf-8" karakterlerin (varsa sembollerin) bozulmasını önler.
        with open(output_txt_path, "w", encoding="utf-8") as file:
            file.write(text)
            
        print(f"\n💾 Metin başarıyla dosyaya kaydedildi: {output_txt_path}")