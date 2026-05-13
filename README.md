# Hasarlı Metinlerin Bilgi Kaybının Giderilmesi (Document Restoration Project)

Bu proje, tarihi ve hasarlı belgelerin restorasyonu için **GAN (Üretken Çekişmeli Ağlar)**, **OCR (Optik Karakter Tanıma)** ve **RAG (Retrieval-Augmented Generation)** teknolojilerini bir araya getiren hibrit bir yapay zeka boru hattı (pipeline) sunmaktadır.

## 🚀 Proje Genel Akışı (Pipeline)

Sistem üç temel aşamadan oluşmaktadır:
1. **Görsel Restorasyon (GAN):** Hasarlı görüntüdeki gürültüler temizlenir ve yapısal bütünlük sağlanır.
2. **Metin Çıkarımı (OCR):** Temizlenmiş görüntüden dijital metin elde edilir.
3. **Anlamsal Restorasyon (RAG):** OCR çıktısındaki harf/kelime hataları, harici bir tarihsel külliyattan (Gutenberg Arşivi) beslenen Llama 3.1 modeli ile düzeltilir.


## 📊 Başarı Metrikleri

Proje genelinde elde edilen performans skorları şu şekildedir:

| Katman | Metrik | Değer |
| :--- | :--- | :--- |
| **GAN** | PSNR / SSIM | 24.67 dB / 0.9818 |
| **OCR** | CER / WER | %14.41 / %47.89 |
| **RAG** | ROUGE-L | %80.56 |
| **RAG** | Cosine Similarity | %93.05 |

## 🧠 Tartışma ve "Metrik Yanılsaması" (Ablation Study)

Yapılan analizlerde, nicel metriklerin (ROUGE, Cosine) çok yüksek çıkmasına rağmen, modelin metin sonlarında **"halüsinasyon"** ürettiği ve bağlamdan koptuğu tespit edilmiştir. Bu durum, literatürde **"Lost in the Middle"** (Liu vd., 2024) olarak bilinen ve LLM'lerin uzun bağlamların sonlarına doğru dikkatini kaybetmesi (Attention Decay) problemiyle birebir örtüşmektedir.

## 📂 Veri Seti (Dataset)

Bu projede kullanılan eğitim ve test verilerine (512x512 görsel çiftleri ve metin külliyatı) aşağıdaki bağlantıdan ulaşabilirsiniz:

🔗 **[Google Drive - Veri Seti Bağlantısı](https://drive.google.com/file/d/1odFgPuYmtLM92FnVqfm1_S4y1ZohiTOC/view?usp=sharing)**

## 🛠️ Kurulum ve Kullanım

1. Depoyu klonlayın.
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt
