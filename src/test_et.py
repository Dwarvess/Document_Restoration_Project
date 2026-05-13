import sys
try:
    import langchain
    import langchain_community
    print("--------------------------------------------------")
    print(f"PYTHON SURUMU: {sys.version}")
    print(f"LANGCHAIN DOSYASI: {langchain.__file__}")
    print(f"LANGCHAIN VERSION: {langchain.__version__}")
    print(f"COMMUNITY DOSYASI: {langchain_community.__file__}")
    print("--------------------------------------------------")
    from langchain.retrievers import EnsembleRetriever
    print("BASARILI! EnsembleRetriever yüklendi.")
except ImportError as e:
    print(f"HATA DETAYI: {e}")
except Exception as e:
    print(f"GENEL HATA: {e}")