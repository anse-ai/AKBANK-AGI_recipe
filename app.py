from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset
import os
import streamlit as st

# LangChain (new ecosystem, chains-free)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS



# --------------------------------------------------------------------
# Safe Dataset Loader: script dataset + direct parquet/json fallback
# --------------------------------------------------------------------
from datasets import load_dataset

def safe_load_recipe_dataset():
    """Try multiple ways to load the dataset:
       1) script (datasets 2.x + trust_remote_code)
       2) direct parquet/json/csv from repo via hf_hub_download
    """
    repo_id = "m3hrdadfi/recipe_nlg_lite"

    # 1) Script yükleme (datasets==2.x ile)
    try:
        return load_dataset(repo_id, trust_remote_code=True)
    except Exception as e:
        print("Script yükleme başarısız -> dosyadan yüklemeye geçiliyor:", e)

    # 2) Repo dosyalarını listele ve indir (öncelik: parquet > json/jsonl > csv)
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    # olası uzantılar
    candidates = [f for f in files if f.lower().endswith((".parquet", ".jsonl", ".json", ".csv"))]

    if not candidates:
        raise RuntimeError("Repo içinde yüklenebilir veri dosyası (.parquet/.json/.jsonl/.csv) bulunamadı.")

    # indirme ve yükleme denemeleri
    for fname in candidates:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".parquet":
                ds = load_dataset("parquet", data_files=local_path)
            elif ext in (".json", ".jsonl"):
                # jsonl ihtimali için lines=True otomatik anlaşılır; gerekirse aşağıdaki gibi verilebilir:
                ds = load_dataset("json", data_files=local_path)
            elif ext == ".csv":
                ds = load_dataset("csv", data_files=local_path)
            else:
                continue  # bilinmeyen uzantı

            # Tek split gelirse 'train' anahtarı olsun
            if not isinstance(ds, dict):
                ds = {"train": ds}

            return ds
        except Exception as e:
            print(f"Dosyadan yükleme denemesi başarısız ({fname}):", e)

    raise RuntimeError("Recipe dataset yüklenemedi (script + dosyadan yükleme denemeleri başarısız).")

# -----------------------------
# Config & Page
# -----------------------------
st.set_page_config(page_title="Tarif Asistanı (RAG)", page_icon="🍳", layout="wide")
st.title("🍳 Tarif Asistanı — RAG ile Tarif Bulucu (chains-free)")
st.write("Elindeki malzemeleri yaz, veri tabanındaki tariflerden uygun olanları bulup özetleyeyim.")

# Read API key (Streamlit secrets or env var)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google Gemini API anahtarı bulunamadı. Lütfen `.streamlit/secrets.toml` içine `GOOGLE_API_KEY` ekleyin veya ortam değişkeni olarak ayarlayın.")
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def recipe_to_doc(example: dict) -> Document:
    """Convert a dataset row to a LangChain Document."""
    name = (example.get("name") or "").strip()
    desc = (example.get("description") or "").strip()
    ings = (example.get("ingredients") or "").strip()
    steps = (example.get("steps") or "").strip()
    link = (example.get("link") or "").strip()

    text = f"""Title: {name}
Ingredients: {ings}
Steps: {steps}
Description: {desc}"""
    return Document(page_content=text, metadata={"source_link": link, "title": name})


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource(show_spinner="Veri seti indiriliyor ve indeks oluşturuluyor... (ilk çalıştırmada biraz sürebilir)")
def build_retriever(k: int = 4):
    # 1) Load dataset
    ds = safe_load_recipe_dataset()
    train = ds["train"]

    # 2) Convert to Documents
    docs = [recipe_to_doc(row) for row in train]

    # 3) Embeddings & Vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4) Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


@st.cache_resource(show_spinner=False)
def build_llm_and_prompt():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(
        """Sen bir **Tarif Asistanı**sın. Kullanıcının sorusunu ve aşağıdaki tarif içeriklerini kullanarak,
kısa, net ve uygulanabilir bir yanıt üret. Gerekirse malzeme listesi ve adım adım talimat ver.
Yanıtın Türkçe olsun. Uymayan veya eksik tarif varsa bahsetme.

# Kullanıcı Sorusu
{input}

# İlgili Tarif İçerikleri
{context}
"""
    )
    parser = StrOutputParser()
    return llm, prompt, parser


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    top_k = st.slider("Kaç tariften bağlam getirilsin? (k)", min_value=2, max_value=10, value=4, step=1)

# Build once / cached
retriever = build_retriever(k=top_k)
llm, prompt, parser = build_llm_and_prompt()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input
query = st.text_input("Elindeki malzemeleri veya yapmak istediğin yemeği yaz (örn. 'tavuk, biber, domates ile ne yapabilirim?')")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Answer when the last message is user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_q = st.session_state.messages[-1]["content"]
    with st.spinner("Tarif aranıyor ve yanıt hazırlanıyor..."):
        # Retrieve docs (no chains module)
        docs = retriever.invoke(user_q) 
        context_text = format_docs(docs)

        # LCEL pipeline: prompt -> llm -> parser
        answer = (prompt | llm | parser).invoke({"input": user_q, "context": context_text}).strip()

        # Sources
        sources = []
        for d in docs:
            link = d.metadata.get("source_link")
            title = d.metadata.get("title") or "Kaynak"
            if link:
                sources.append(f"- [{title}]({link})")

        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(sorted(set(sources)))

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
