
# Akbank GenAI Bootcamp - RAG Recipe Chatbot

Bu proje, **Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı** [cite: 1] kapsamında geliştirilmiştir. [cite\_start]Projenin ana hedefi, Retrieval-Augmented Generation (RAG) [cite: 2] mimarisini kullanarak bir chatbot oluşturmak ve bu chatbot'u bir web arayüzü üzerinden sunmaktır.

## Projenin Amacı 

Bu projenin amacı, kullanıcılara yemek tarifleri konusunda yardımcı olan bir "Tarif Asistanı" chatbot'u oluşturmaktır. Chatbot, kullanıcıların "Elimde tavuk, tuz ve biber var, ne yapabilirim?" gibi sorularına, sahip olduğu tarif veri setinden ilgili tarifleri bularak ve bu bilgileri kullanarak cevaplar üretebilir.

## Çözüm Mimarisi (Kullanılan Yöntemler)

Proje, temel olarak bir RAG (Retrieval-Augmented Generation) [cite: 23] pipeline'ı üzerine kuruludur. Kullanılan teknolojiler ve mimari bileşenleri proje gereksinimlerinde belirtilen örneklere göre seçilmiştir:

  * **RAG Framework:** **LangChain** 
      * Tüm bileşenleri (veri yükleyici, vektör deposu, retriever, LLM ve prompt) bir araya getiren ana kütüphanedir.
  * **Generation Model (LLM):** **Gemini 1.5 Flash** (Google) 
      * Kullanıcının sorusuna ve bulunan tariflere (context) göre nihai cevabı üreten üretken yapay zeka modelidir.
  * **Embedding Model:** **Google `models/embedding-001`**
      * Hem tarif veri setindeki tarifleri hem de kullanıcının sorusunu sayısal vektörlere dönüştürmek için kullanılan modeldir.
  * **Vector Database:** **FAISS** (Facebook AI) 
      * Tüm tarif vektörlerini saklayan ve anlamsal olarak benzer olanları hızla bulmayı (retrieval) sağlayan yerel (local) bir vektör veritabanıdır.
  * **Web Arayüzü:** **Streamlit**
      * Frontend bilgisi gerektirmeden, Python ile hızlıca interaktif bir web arayüzü oluşturmak için kullanılmıştır.

## Veri Seti Hakkında Bilgi 

Projede hazır bir veri seti kullanılmıştır.

  * **Veri Seti:** `m3hrdadfi/recipe_nlg_lite`
  * **Kaynak:** Hugging Face Datasets
  * **İçerik:** Bu veri seti 7,000'den fazla yemek tarifi içermektedir. Her tarif için `name` (tarif adı), `ingredients` (malzemeler) ve `steps` (hazırlanış adımları) gibi temel bilgiler bulunmaktadır.
  * **Kullanım Metodolojisi:** Veri seti yüklendikten sonra, her tarif "Tarif Adı / Malzemeler / Hazırlanışı" formatında tek bir metin belgesine dönüştürülmüş ve FAISS vektör veritabanında indekslenmiştir.

## Kodun Çalışma Kılavuzu 

Bu projenin yerel makinenizde (lokal) veya bir bulut platformunda çalıştırılması için aşağıdaki adımlar izlenmelidir.

1.  **Repository'i Klonlayın:**

    ```bash
    git clone https://github.com/anse-ai/AKBANK-AGI_recipe.git
    cd AKBANK-AGI_recipe
    ```

2.  **Sanal Ortam Kurulumu (Önerilen):** 

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Gerekli Kütüphanelerin Kurulumu:**
    Projenin tüm bağımlılıkları `requirements.txt` dosyasında listelenmiştir.

    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarının Ayarlanması (Secrets):**
    Projenin Google Gemini API'yi kullanabilmesi için bir API anahtarına ihtiyacı vardır.

      * Proje ana dizininde `.streamlit` adında bir klasör oluşturun.
      * İçine `secrets.toml` adında bir dosya oluşturun.
      * Dosyanın içine Google AI Studio'dan aldığınız API anahtarınızı aşağıdaki gibi ekleyin:
        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "YOUR_API_KEY_GOES_HERE"
        ```

5.  **Uygulamayı Çalıştırma:** 

    ```bash
    streamlit run app.py
    ```

## Elde Edilen Sonuçlar & Product Kılavuzu 

Sonuç olarak, kullanıcıların sorduğu sorulara ilgili tarif veri setinden (knowledge base) bilgiler bularak cevap verebilen bir RAG tabanlı chatbot geliştirilmiştir.

### 🌐 Live Demo Linki

> **[BURAYA DEPLOY ETTİĞİNİZ LİNKİ EKLEYİN (örn: Hugging Face Spaces, Streamlit Cloud, vb.)]**

### Ekran Görüntüleri 

Uygulamanın nasıl çalıştığını gösteren örnek ekran görüntüleri:

**(Buraya uygulamanızın bir ekran görüntüsünü ekleyin)**

**(Buraya chatbot'un bir soruya verdiği cevabın ekran görüntüsünü ekleyin)**

(End of `README.md` file)
