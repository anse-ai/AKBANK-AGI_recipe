from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset
import os
import streamlit as st
import logging, warnings

# LangChain (chains-free)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.globals import set_verbose
set_verbose(False)

# Hybrid retrieval
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- HF uyarılarını kıs ---
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from datasets.utils.logging import set_verbosity
from datasets.utils.logging import ERROR as HF_ERROR
set_verbosity(HF_ERROR)
warnings.filterwarnings("ignore", message="Repo card metadata block was not found")

# ==========================
# HIZ / GENEL AYARLAR
# ==========================
TOP_K_DEFAULT   = 4           # NEGATİF OLMAMALI!
MAX_OUTPUT_TOK  = 384
SAMPLE_N        = 5000        # ilk kez indekste hız için örneklem; tam veri istersen None
PERSIST_DIR     = "chroma_db"  # indeksi diske yaz, sonraki açılışlar hızlı
# NotFound yaşamamak için: v1beta + model adı -001'siz
MODEL_NAME      = "gemini-2.5-flash"
API_VERSION     = "v1beta"

# -----------------------------
# Safe Dataset Loader
# -----------------------------
def safe_load_recipe_dataset():
    repo_id = "m3hrdadfi/recipe_nlg_lite"
    try:
        return load_dataset(repo_id, trust_remote_code=True)
    except Exception as e:
        print("Script yükleme başarısız -> dosyadan yüklemeye geçiliyor:", e)

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [f for f in files if f.lower().endswith((".parquet", ".jsonl", ".json", ".csv"))]
    if not candidates:
        raise RuntimeError("Repo içinde yüklenebilir veri (.parquet/.json/.jsonl/.csv) bulunamadı.")

    for fname in candidates:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".parquet":
                ds = load_dataset("parquet", data_files=local_path)
            elif ext in (".json", ".jsonl"):
                ds = load_dataset("json", data_files=local_path)
            elif ext == ".csv":
                ds = load_dataset("csv", data_files=local_path)
            else:
                continue
            if not isinstance(ds, dict):
                ds = {"train": ds}
            return ds
        except Exception as e:
            print(f"Dosyadan yükleme başarısız ({fname}):", e)

    raise RuntimeError("Recipe dataset yüklenemedi (script + dosyadan yükleme denemeleri başarısız).")

# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Tarif Asistanı (RAG)", page_icon="🍳", layout="wide")
st.title("🍳 Tarif Asistanı — hızlı RAG")
st.write("Elindeki malzemeleri yaz; uygun tarifleri bulup özetleyeyim.")

# API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google Gemini API anahtarı bulunamadı. `.streamlit/secrets.toml` içine `GOOGLE_API_KEY` ekleyin veya ortam değişkeni olarak ayarlayın.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def recipe_to_doc(example: dict) -> Document:
    name = (example.get("name") or "").strip()
    ings = (example.get("ingredients") or "").strip()
    steps = (example.get("steps") or "").strip()
    link = (example.get("link") or "").strip()
    # Hız için description’ı atladık; bağlamı kısa tut
    text = f"Title: {name}\nIngredients: {ings}\nSteps: {steps}"
    return Document(page_content=text, metadata={"source_link": link, "title": name})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def normalize_ingredients(text: str) -> list[str]:
    seps = [",", ";", "/", "+", "|", "&"]
    t = text.lower().strip()
    for s in seps:
        t = t.replace(s, " ")
    t = t.replace(" ve ", " ")
    tokens = [tok.strip() for tok in t.split() if tok.strip()]
    return tokens

# TR ↔ EN eşlemeler (kapsamı artırmak için)
ING_SYNONYMS = {
    "biber": ["pepper", "bell pepper", "capsicum", "chili", "chilli"],
    "domates": ["tomato", "tomatoes"],
    "patlıcan": ["eggplant", "aubergine"],
    "kabak": ["zucchini", "courgette", "squash"],
    "maydanoz": ["parsley"],
    "kıyma": ["minced beef", "ground beef", "minced meat", "ground meat"],
    "tavuk": ["chicken"],
    "süt": ["milk"],
    "yoğurt": ["yogurt", "yoghurt"],
    "tereyağı": ["butter"],
    "peynir": ["cheese"],
    "soğan": ["onion"],
    "sarımsak": ["garlic"],
    "pirinç": ["rice"],
    "bulgur": ["bulgur", "cracked wheat"],
    "yufka": ["phyllo", "filo"],
    "un": ["flour"],
    "şeker": ["sugar"],
    "domates salçası": ["tomato paste"],
    "biber salçası": ["pepper paste"],
}

def expand_query_with_synonyms(user_q: str) -> str:
    toks = normalize_ingredients(user_q)
    expanded = set(toks)
    for tok in toks:
        if tok in ING_SYNONYMS:
            expanded.update(ING_SYNONYMS[tok])
    return " ".join(sorted(expanded))

# -----------------------------
# CACHE: dataset -> rows
# -----------------------------
@st.cache_data(show_spinner="Dataset yükleniyor...")
def load_rows(sample_n=SAMPLE_N):
    ds = safe_load_recipe_dataset()
    train = ds["train"]
    if sample_n and len(train) > sample_n:
        train = train.select(range(sample_n))
    return train.to_list()

# -----------------------------
# CACHE: embeddings + vectorstore (persist) + ENSEMBLE
# -----------------------------
def _safe_k(k: int, minimum: int = 1) -> int:
    try:
        k = int(k)
    except Exception:
        k = minimum
    return max(minimum, k)

@st.cache_resource(show_spinner="Vektör indeksi hazırlanıyor...")
def build_retriever(k: int = TOP_K_DEFAULT):
    # k'yi baştan güvene al
    k = _safe_k(k, minimum=2)

    rows = load_rows()
    docs = [recipe_to_doc(r) for r in rows]

    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )

    # Chroma store (persist)
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)

    # Koleksiyon boyutuna göre güvenli k
    try:
        coll_count = vectorstore._collection.count()
    except Exception:
        coll_count = None
    k_chroma = _safe_k(min(k, coll_count) if coll_count is not None else k, minimum=1)

    # Chroma retriever (semantik)
    chroma_ret = vectorstore.as_retriever(search_kwargs={"k": k_chroma})

    # BM25 retriever (anahtar kelime)
    bm25_ret = BM25Retriever.from_documents(docs)
    bm25_ret.k = _safe_k(min(max(2 * k, 6), len(docs)), minimum=1)

    # Ensemble (birleştir)
    ensemble_top_k = _safe_k(min(k, len(docs)), minimum=1)
    ensemble = EnsembleRetriever(
        retrievers=[chroma_ret, bm25_ret],
        weights=[0.6, 0.4],
        top_k=ensemble_top_k,
    )
    return ensemble

# -----------------------------
# CACHE: LLM + Prompt + Parser
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_llm_and_prompt():
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_version=API_VERSION,   # v1beta + -001'siz model → 404 yok
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOK,
        google_api_key=GOOGLE_API_KEY,
        streaming=False,           # istersen True yap; paket setin desteklemeli
    )
    prompt = ChatPromptTemplate.from_template(
        "Sen kısa ve net bir Tarif Asistanısın. Soruyu ve bağlamı kullanarak Türkçe, öz bir cevap ver.\n\n"
        "Soru:\n{input}\n\nBağlam:\n{context}\n"
    )
    parser = StrOutputParser()
    return llm, prompt, parser

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    top_k = st.slider("Kaç tariften bağlam (k)?", min_value=2, max_value=8, value=TOP_K_DEFAULT, step=1)
    st.caption(f"Max yanıt token: {MAX_OUTPUT_TOK} • Örneklem: {SAMPLE_N or 'TAM'} • Persist: {PERSIST_DIR}")

retriever = build_retriever(k=top_k)
llm, prompt, parser = build_llm_and_prompt()

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Elindeki malzemeleri veya yapmak istediğin yemeği yaz (örn. 'tavuk, biber, domates...')")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# Yanıt (invoke)
# -----------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_q = st.session_state.messages[-1]["content"]
    with st.spinner("Tarif aranıyor ve yanıt hazırlanıyor..."):
        aug_q = expand_query_with_synonyms(user_q)

        docs = retriever.invoke(aug_q)

        # Hiç doküman dönmezse kademeli fallback
        if not docs:
            try:
                docs = retriever.invoke(aug_q + " recipe ingredients steps")
            except Exception:
                pass

        if not docs:
            rows = load_rows()
            toks = normalize_ingredients(user_q)
            hits = []
            for r in rows:
                blob = " ".join([(r.get("name") or ""), (r.get("ingredients") or ""), (r.get("steps") or "")]).lower()
                if len(toks) == 1:
                    if toks[0] in blob:
                        hits.append(recipe_to_doc(r))
                else:
                    if all(t in blob for t in toks[:2]):
                        hits.append(recipe_to_doc(r))
                if len(hits) >= 5:
                    break
            docs = hits or []

        context_text = format_docs(docs)
        chain = (prompt | llm | parser)

        try:
            final_text = chain.invoke({"input": user_q, "context": context_text}).strip()
        except Exception as e:
            final_text = f"Üretim sırasında bir sorun oluştu: {e}"

        # Kaynaklar
        sources = []
        for d in docs:
            link = d.metadata.get("source_link")
            title = d.metadata.get("title") or "Kaynak"
            if link:
                sources.append(f"- [{title}]({link})")
        sources_block = "\n\n**Sources:**\n" + "\n".join(sorted(set(sources))) if sources else ""
        final_text += sources_block

    st.session_state.messages.append({"role": "assistant", "content": final_text})
    st.chat_message("assistant").write(final_text)
