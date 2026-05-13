import os

# DİKKAT: Buraya kendi Kaggle_Yukleme klasörünün tam yolunu yaz
hedef_klasor = r"D:\Document_Restoration_Project\data"
degisen_sayisi = 0

print("Tarama başlatıldı, lütfen bekleyin...")

# Klasörün içindeki tüm dosyaları alt klasörleriyle birlikte tara
for kok_dizin, alt_klasorler, dosyalar in os.walk(hedef_klasor):
    for dosya in dosyalar:
        if "'" in dosya: # Eğer dosya adında kesme işareti varsa
            eski_yol = os.path.join(kok_dizin, dosya)
            yeni_dosya_adi = dosya.replace("'", "") # Kesme işaretini sil
            yeni_yol = os.path.join(kok_dizin, yeni_dosya_adi)
            
            os.rename(eski_yol, yeni_yol)
            degisen_sayisi += 1

print(f"✅ İşlem tamam! Toplam {degisen_sayisi} dosyanın adı düzeltildi.")