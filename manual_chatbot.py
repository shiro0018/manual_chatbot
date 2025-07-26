import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from pathlib import Path
import json

load_dotenv()

st.set_page_config(page_title="マニュアルチャットボット", layout="wide")
st.title("📘 マニュアルチャットボット")

VECTOR_DIR = "vectorstore"
LABEL_FILE = os.path.join(VECTOR_DIR, "labels.json")
os.makedirs(VECTOR_DIR, exist_ok=True)

def load_labels():
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

labels = load_labels()

with st.expander("📤 PDFマニュアルのアップロード｜ファイル名は半角英数で", expanded=False):
    uploaded_file = st.file_uploader("PDFファイルを選択", type=["pdf"])
    display_name = st.text_input("画面表示用マニュアル名（例：休暇申請マニュアル）")

    if st.button("📚 読み込む"):
        if not uploaded_file or not display_name:
            st.warning("📄 ファイルと表示名の両方を入力してください")
        elif uploaded_file.size > 5 * 1024 * 1024:
            st.error("❌ ファイルサイズが大きすぎます（5MBまで）")
        else:
            safe_filename = os.path.basename(uploaded_file.name)
            filename = Path(safe_filename).stem
            pdf_path = os.path.join(VECTOR_DIR, safe_filename)
            persist_dir = os.path.join(VECTOR_DIR, filename)

            try:
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings()
                db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
                db.persist()

                labels[filename] = display_name
                save_labels(labels)

                st.success("✅ 登録が完了しました！")

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 📄 マニュアル一覧")
    with st.container():
        for vec_dir in sorted(Path(VECTOR_DIR).glob("*")):
            if vec_dir.is_dir() and (vec_dir / "chroma.sqlite3").exists():
                key = vec_dir.name
                label = labels.get(key, key)
                st.markdown(
                    f"<div style='font-size: 0.85em; margin-bottom: 0.5em;'>📘 <b>{label}</b></div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"選択", key=key):
                    st.session_state.selected_manual = key

with col2:
    if "selected_manual" in st.session_state:
        selected = st.session_state.selected_manual
        label = labels.get(selected, selected)

        st.markdown(f"#### ✅ 現在選択中のマニュアル：{label}")
        prompt_text = f"💬 「{label}」について質問を入力してください:"
        question = st.text_input(prompt_text, key="question_input")

        if question:
            try:
                with st.spinner("回答を生成中..."):
                    embeddings = OpenAIEmbeddings()
                    db = Chroma(persist_directory=os.path.join(VECTOR_DIR, selected), embedding_function=embeddings)
                    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
                    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
                    answer = qa.run(question)
                    st.write("🧠 回答:")
                    st.success(answer)
            except Exception as e:
                st.error(f"❌ 回答の生成中にエラーが発生しました: {e}")
