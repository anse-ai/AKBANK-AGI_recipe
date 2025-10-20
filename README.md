# AKBANK-AGI_recipe
 Yeni Nesil Proje Kampı  kapsamında geliştirilmiştir. Projenin ana hedefi, Retrieval-Augmented Generation (RAG)  mimarisini kullanarak bir chatbot oluşturmak ve bu chatbot'u bir web arayüzü üzerinden sunmaktır.


This is the final and most important step to complete your project. [cite\_start]This `README.md` file combines all the project steps into one document as required by the bootcamp[cite: 8].

You just need to create the files in your GitHub repository.

-----

### **Your GitHub Repository Structure**

Your repository should look like this:

```
/
├── .streamlit/
│   └── secrets.toml     (This contains your API key)
├── app.py               (The Streamlit app code)
├── requirements.txt     (The list of libraries)
└── README.md            (The file we are writing now)
```

-----

### **The `README.md` File**

Copy and paste the text below into the `README.md` file in your repository. **You will need to add your own deployment link and screenshots** once you have them.

-----

(Start of `README.md` file)

# Akbank GenAI Bootcamp - RAG Recipe Chatbot

[cite\_start]Bu proje, **Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı** [cite: 1] kapsamında geliştirilmiştir. [cite\_start]Projenin ana hedefi, Retrieval-Augmented Generation (RAG) [cite: 2] mimarisini kullanarak bir chatbot oluşturmak ve bu chatbot'u bir web arayüzü üzerinden sunmaktır.

## [cite\_start]Projenin Amacı [cite: 9]

Bu projenin amacı, kullanıcılara yemek tarifleri konusunda yardımcı olan bir "Tarif Asistanı" chatbot'u oluşturmaktır. Chatbot, kullanıcıların "Elimde tavuk, tuz ve biber var, ne yapabilirim?" gibi sorularına, sahip olduğu tarif veri setinden ilgili tarifleri bularak ve bu bilgileri kullanarak cevaplar üretebilir.

## [cite\_start]Çözüm Mimarisi (Kullanılan Yöntemler) [cite: 11, 23]

[cite\_start]Proje, temel olarak bir RAG (Retrieval-Augmented Generation) [cite: 23] pipeline'ı üzerine kuruludur. [cite\_start]Kullanılan teknolojiler ve mimari bileşenleri proje gereksinimlerinde belirtilen örneklere göre seçilmiştir[cite: 42, 43, 44]:

  * [cite\_start]**RAG Framework:** **LangChain** [cite: 44]
      * Tüm bileşenleri (veri yükleyici, vektör deposu, retriever, LLM ve prompt) bir araya getiren ana kütüphanedir.
  * [cite\_start]**Generation Model (LLM):** **Gemini 1.5 Flash** (Google) [cite: 42]
      * Kullanıcının sorusuna ve bulunan tariflere (context) göre nihai cevabı üreten üretken yapay zeka modelidir.
  * [cite\_start]**Embedding Model:** **Google `models/embedding-001`** [cite: 43]
      * Hem tarif veri setindeki tarifleri hem de kullanıcının sorusunu sayısal vektörlere dönüştürmek için kullanılan modeldir.
  * [cite\_start]**Vector Database:** **FAISS** (Facebook AI) [cite: 43]
      * Tüm tarif vektörlerini saklayan ve anlamsal olarak benzer olanları hızla bulmayı (retrieval) sağlayan yerel (local) bir vektör veritabanıdır.
  * **Web Arayüzü:** **Streamlit**
      * Frontend bilgisi gerektirmeden, Python ile hızlıca interaktif bir web arayüzü oluşturmak için kullanılmıştır.

## [cite\_start]Veri Seti Hakkında Bilgi [cite: 10]

[cite\_start]Projede hazır bir veri seti kullanılmıştır[cite: 17].

  * **Veri Seti:** `m3hrdadfi/recipe_nlg_lite`
  * **Kaynak:** Hugging Face Datasets
  * **İçerik:** Bu veri seti 7,000'den fazla yemek tarifi içermektedir. Her tarif için `name` (tarif adı), `ingredients` (malzemeler) ve `steps` (hazırlanış adımları) gibi temel bilgiler bulunmaktadır.
  * **Kullanım Metodolojisi:** Veri seti yüklendikten sonra, her tarif "Tarif Adı / Malzemeler / Hazırlanışı" formatında tek bir metin belgesine dönüştürülmüş ve FAISS vektör veritabanında indekslenmiştir.

## [cite\_start]Kodun Çalışma Kılavuzu [cite: 20]

Bu projenin yerel makinenizde (lokal) veya bir bulut platformunda çalıştırılması için aşağıdaki adımlar izlenmelidir.

1.  **Repository'i Klonlayın:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  [cite\_start]**Sanal Ortam Kurulumu (Önerilen):** [cite: 21]

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Gerekli Kütüphanelerin Kurulumu:**
    [cite\_start]Projenin tüm bağımlılıkları `requirements.txt` dosyasında listelenmiştir[cite: 21].

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

5.  [cite\_start]**Uygulamayı Çalıştırma:** [cite: 21]

    ```bash
    streamlit run app.py
    ```

## [cite\_start]Elde Edilen Sonuçlar & Product Kılavuzu [cite: 12, 25]

Sonuç olarak, kullanıcıların sorduğu sorulara ilgili tarif veri setinden (knowledge base) bilgiler bularak cevap verebilen bir RAG tabanlı chatbot geliştirilmiştir.

### [cite\_start]🌐 Live Demo Linki [cite: 13]

> **[BURAYA DEPLOY ETTİĞİNİZ LİNKİ EKLEYİN (örn: Hugging Face Spaces, Streamlit Cloud, vb.)]**

### [cite\_start]Ekran Görüntüleri [cite: 25]

Uygulamanın nasıl çalıştığını gösteren örnek ekran görüntüleri:

**(Buraya uygulamanızın bir ekran görüntüsünü ekleyin)**

**(Buraya chatbot'un bir soruya verdiği cevabın ekran görüntüsünü ekleyin)**

(End of `README.md` file)
