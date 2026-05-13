import os
import glob
import nltk
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from spellchecker import SpellChecker

# NLTK cümle tokenize edicilerini indir (bir kereye mahsus)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_gutenberg_text(text):
    """
    Project Gutenberg metinlerindeki başlık/lisans kısımlarını temizler.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if start_marker in text and end_marker in text:
        text = text.split(start_marker)[1].split(end_marker)[0]
    return text.strip()


def create_sentence_aware_knowledge_base(document_paths, persist_dir="./chroma_db_sentences"):
    """
    Metinleri anlamlı cümlelere bölerek bir vektör veritabanı oluşturur.
    Gutenberg temizleme işlemi otomatik uygulanır.
    """
    all_sentences = []
    for path in document_paths:
        print(f"İşleniyor: {os.path.basename(path)}")
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        cleaned_text = clean_gutenberg_text(raw_text)
        sentences = sent_tokenize(cleaned_text)

        for sentence in sentences:
            if len(sentence.strip()) > 10:
                all_sentences.append(Document(page_content=sentence.strip()))

    print(f"\nToplam {len(all_sentences)} adet anlamlı cümle oluşturuldu.")
    print("Embedding modeli yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Vektör veritabanı oluşturuluyor (bu işlem biraz zaman alabilir)...")
    vectorstore = Chroma.from_documents(
        documents=all_sentences,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"\n✅ Cümle-bazlı bilgi tabanı '{persist_dir}' konumuna kaydedildi.")
    return vectorstore


def load_knowledge_base(persist_dir="./chroma_db_sentences"):
    """Daha önce oluşturulmuş vektör veritabanını yükler."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectorstore


def build_knowledge_base_from_folder(folder_path="data/raw_books", persist_dir="./chroma_db_sentences"):
    """
    Belirtilen klasördeki tüm .txt dosyalarını bulur ve bilgi tabanını oluşturur.
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not txt_files:
        print(f"HATA: '{folder_path}' içinde hiç .txt dosyası bulunamadı!")
        return None

    print(f"İşlenecek {len(txt_files)} adet kitap bulundu:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")

    return create_sentence_aware_knowledge_base(txt_files, persist_dir=persist_dir)


def restore_text_with_transparency(damaged_text, vectorstore, model_name="llama3.1:8b", similarity_threshold=0.99):
    """
    Hasarlı metni, yazım ön işleme ve benzerlik skoru kontrolü ile düzeltir.
    """
    # --- Yazım Denetimi Ön İşlemi ---
    spell = SpellChecker(language='en')
    words = damaged_text.split()
    corrected_words = []
    for word in words:
        # Alfabetik kısmı ayıkla
        clean_word = ''.join(filter(str.isalpha, word))
        if clean_word and clean_word not in spell:
            corrected = spell.correction(clean_word)
            if corrected:
                word = word.replace(clean_word, corrected)
        corrected_words.append(word)
    preprocessed_text = ' '.join(corrected_words)
    print(f"\n🔧 Yazım ön işleme: '{damaged_text[:50]}...' -> '{preprocessed_text[:50]}...'")

    # Genişletilmiş retrieval (k=5)
    results_with_scores = vectorstore.similarity_search_with_score(preprocessed_text, k=5)

    relevant_docs = []
    print("\n--- Retrieval Diagnostics ---")
    for doc, score in results_with_scores:
        similarity = 1 - score
        print(f"Skor: {score:.4f} | Benzerlik: {similarity:.4f} | İçerik: {doc.page_content[:100]}...")
        if similarity > similarity_threshold:
            relevant_docs.append(doc)

    if relevant_docs:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"\n✅ Eşik değeri ({similarity_threshold}) aşıldı. {len(relevant_docs)} adet belge kullanılıyor.")
        prompt = f"""
        You are a text restoration assistant. A damaged OCR output or incomplete English text is provided below.
        Use the following context to correct the text and make it semantically coherent.

        Context:
        {context}

        Damaged text:
        "{damaged_text}"

        Please provide only the corrected version of the text, without any additional explanation or commentary.
        """
    else:
        print(f"\n⚠️ Hiçbir belge eşik değerini ({similarity_threshold}) geçemedi. LLM kendi bilgisiyle düzeltecek.")
        prompt = f"""
        You are a text restoration assistant. A damaged OCR output or incomplete English text is provided below.
        Unfortunately, no relevant context was found in the knowledge base. Please correct the text to the best of your general knowledge.

        Damaged text:
        "{damaged_text}"

        Please provide only the corrected version of the text, without any additional explanation or commentary.
        """

    llm = Ollama(model=model_name, temperature=0)
    response = llm.invoke(prompt)
    return response.strip()


# --- İNTERAKTİF TEST ARAYÜZÜ ---
# --- TAM OTOMATİK BORU HATTI (PIPELINE) ARAYÜZÜ ---
if __name__ == "__main__":
    print("=== Tam Otomatik RAG Restorasyon (GAN -> OCR -> RAG) ===")

    # Bilgi tabanını yükle
    try:
        kb = load_knowledge_base("./chroma_db_sentences")
        print("✅ Bilgi tabanı başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Bilgi tabanı yüklenemedi: {e}")
        print("Lütfen önce bilgi tabanını oluşturun.")
        exit()

    # 1. OCR'ın Kaydettiği Dosyayı Oku
    ocr_txt_path = r"D:\Document_Restoration_Project\outputs\ocr_result.txt"

    if not os.path.exists(ocr_txt_path):
        print(f"HATA: OCR dosyası bulunamadı! Lütfen önce OCR adımını çalıştırın.\nAranan Yol: {ocr_txt_path}")
        exit()

    with open(ocr_txt_path, "r", encoding="utf-8") as file:
        damaged_text = file.read().strip()

    if not damaged_text:
        print("HATA: OCR dosyası boş!")
        exit()

    print("\n📝 [Sisteme Giren Bozuk OCR Metni]:")
    print(f"\033[93m{damaged_text}\033[0m")

    # 2. Gelişmiş RAG ve LLM ile Metni Onar
    print("\n⚙️ [Llama 3.1 Anlamsal Onarımı Başlatıyor...]")
    corrected = restore_text_with_transparency(
        damaged_text=damaged_text,
        vectorstore=kb,
        model_name="llama3.1:8b",
        similarity_threshold=0.99
    )

    # 3. Sonucu Ekrana Yazdır
    print("\n✨ [Onarılmış Kusursuz Metin]:")
    print(f"\033[92m{corrected}\033[0m")

    # 4. Final Metni Kaydet
    final_output_path = r"D:\Document_Restoration_Project\outputs\final_restored.txt"
    with open(final_output_path, "w", encoding="utf-8") as file:
        file.write(corrected)

    print(f"\n💾 İşlem Tamamlandı! Final metin şuraya kaydedildi: {final_output_path}")
    print("-" * 50)