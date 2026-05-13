import glob
import os

def build_knowledge_base_from_folder(folder_path="data/raw_books"):
    # Klasördeki tüm .txt dosyalarını bul
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print(f"HATA: '{folder_path}' içinde hiç .txt dosyası bulunamadı!")
        return
    
    print(f"İşlenecek {len(txt_files)} adet kitap bulundu:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")
    
    # Mevcut fonksiyonunuzu çağırın (create_sentence_aware_knowledge_base)
    create_sentence_aware_knowledge_base(txt_files, persist_dir="./chroma_db_sentences")

# Çalıştır
if __name__ == "__main__":
    build_knowledge_base_from_folder()