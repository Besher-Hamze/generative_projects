"""
مولد مشاريع التخرج - وضع المحادثة (Chat Mode)
يقرأ المشاريع من ملف CSV ويتيح لك المحادثة مع Ollama لاقتراح مشاريع جديدة.
لا يحتاج MongoDB - فقط Ollama والمكتبات المتجهة.
"""

import os
import requests
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==================== الإعدادات ====================
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"
CSV_FILE = "merged.csv"          # ملف البيانات المدخلة
PERSIST_DIR = "./chroma_db_chat" # مجلد حفظ قاعدة البيانات المتجهة
TOP_K = 5                        # عدد المشاريع المشابهة التي يتم جلبها
MAX_HISTORY_TURNS = 3            # عدد الأدوار السابقة في المحادثة التي يتذكرها


# ==================== دوال الاتصال مع Ollama ====================
def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """استدعاء نموذج Ollama للحصول على الرد."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=300,
        )
        return response.json()["message"]["content"]
    except Exception as e:
        return (
            f"❌ خطأ في الاتصال مع Ollama: {str(e)}\n"
            "تأكد من تشغيل Ollama بالأمر: ollama serve"
        )


# ==================== تحميل المشاريع من CSV ====================
def load_projects_from_csv(csv_path: str):
    """
    قراءة ملف CSV وتحويله إلى قائمة من الوثائق (Documents) مناسبة للبحث المتجه.
    الأعمدة المتوقعة: الدكتور المشرف، العام الدراسي، اسم المشروع، وصف المشروع.
    """
    df = pd.read_csv(csv_path)

    # إزالة الأسطر الفارغة تماماً
    df = df.dropna(how="all")
    # إزالة المشاريع بدون اسم أو وصف
    df = df.dropna(subset=["اسم المشروع", "وصف المشروع"], how="all")

    documents = []
    for _, row in df.iterrows():
        project_name = str(row.get("اسم المشروع", "")).strip()
        description = str(row.get("وصف المشروع", "")).strip()
        supervisor = str(row.get("الدكتور المشرف", "")).strip()
        year = str(row.get("العام الدراسي", "")).strip()

        # تخطي الأسطر غير الصالحة
        if not project_name or project_name.lower() == "nan":
            continue

        content = (
            f"اسم المشروع: {project_name}\n"
            f"وصف المشروع: {description}\n"
            f"المشرف: {supervisor}\n"
            f"العام الدراسي: {year}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "project_name": project_name,
                    "year": year,
                    "supervisor": supervisor,
                },
            )
        )

    return documents


# ==================== تهيئة قاعدة البيانات المتجهة ====================
def initialize_vectordb():
    """بناء أو تحميل قاعدة بيانات Chroma من المشاريع."""
    if not os.path.exists(CSV_FILE):
        print(f"❌ ملف {CSV_FILE} غير موجود في المجلد الحالي!")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # إذا كانت قاعدة البيانات موجودة مسبقاً نحمّلها، وإلا ننشئها
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("⏳ جاري تحميل قاعدة البيانات المتجهة الموجودة...")
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        print("⏳ جاري قراءة المشاريع من ملف CSV...")
        documents = load_projects_from_csv(CSV_FILE)
        print(f"✅ تم تحميل {len(documents)} مشروع.")

        if not documents:
            print("❌ لم يتم العثور على أي مشروع صالح في الملف!")
            return None

        print("⏳ جاري إنشاء قاعدة البيانات المتجهة (قد يستغرق دقيقة)...")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )

    return db


# ==================== البحث وتوليد الرد ====================
def search_similar(query: str, db, k: int = TOP_K) -> str:
    """البحث عن أقرب k مشروع للسؤال."""
    if db is None:
        return ""
    results = db.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in results)


def search_projects(query: str, db, k: int = 3) -> list:
    """إرجاع أقرب المشاريع مع تفاصيلها الكاملة بشكل منظم."""
    if db is None:
        return []

    try:
        rows = db.similarity_search_with_score(query, k=k)
        docs_with_scores = rows
    except Exception:
        docs = db.similarity_search(query, k=k)
        docs_with_scores = [(doc, None) for doc in docs]

    projects = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        metadata = doc.metadata or {}
        projects.append(
            {
                "rank": idx,
                "project_name": metadata.get("project_name", ""),
                "description": _extract_field_from_content(doc.page_content, "وصف المشروع:"),
                "supervisor": metadata.get("supervisor", ""),
                "year": metadata.get("year", ""),
                "full_text": doc.page_content,
                "score": score,
            }
        )

    return projects


def _extract_field_from_content(content: str, key: str) -> str:
    """استخراج قيمة سطر يبدأ بمفتاح معين من نص الـ document."""
    for line in content.splitlines():
        if line.startswith(key):
            return line.replace(key, "", 1).strip()
    return ""


def format_history(history: list, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """تحويل آخر عدة أدوار من المحادثة إلى نص يمكن إرساله مع الـ prompt."""
    if not history:
        return ""
    recent = history[-max_turns * 2:]
    lines = ["المحادثة السابقة:"]
    for msg in recent:
        role_ar = "الطالب" if msg["role"] == "user" else "المساعد"
        lines.append(f"{role_ar}: {msg['content']}")
    return "\n".join(lines)


def generate_response(
    user_input: str,
    context: str,
    conversation_history: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """توليد رد المساعد بناءً على السؤال والمشاريع المشابهة وتاريخ المحادثة."""
    prompt = f"""أنت مساعد ذكي متخصص في توليد أفكار مشاريع تخرج في هندسة المعلوماتية والذكاء الاصطناعي.

المشاريع المشابهة من قاعدة البيانات (للاستئناس فقط):
{context}

{conversation_history}

سؤال/طلب الطالب الحالي: {user_input}

المطلوب منك:
- إذا كان الطالب يطلب فكرة مشروع جديد، قم بتوليد فكرة مبتكرة تتضمن:
  1. عنوان المشروع
  2. وصف تفصيلي للمشروع
  3. التقنيات والأدوات المقترحة (مثل: Python, TensorFlow, OpenCV, Flask, ...)
  4. المكتبات المطلوبة
  5. خطوات التنفيذ الأساسية
  6. الفوائد المتوقعة

- إذا كان يسأل عن تفاصيل معينة أو تحسين مشروع، أجب بشكل واضح ومفصل.
- لا تنسخ المشاريع الموجودة حرفياً، بل استلهم منها.

تأكد أن:
✓ المشروع مبتكر وليس نسخة مباشرة
✓ التقنيات حديثة وعملية
✓ الشرح منظم وواضح
✓ الإجابة بالعربية بشكل كامل

الرد:"""
    return call_ollama(prompt, model=model)


# ==================== حلقة المحادثة الرئيسية ====================
def main():
    print("=" * 60)
    print("🎓 مولد مشاريع التخرج - وضع المحادثة")
    print("=" * 60)

    vectordb = initialize_vectordb()
    if vectordb is None:
        print("❌ فشل تحميل قاعدة البيانات. الخروج...")
        return

    print("\n✅ النظام جاهز!")
    print("💡 اكتب سؤالك، أو 'خروج' / 'exit' للإنهاء.")
    print("-" * 60)

    conversation = []

    while True:
        try:
            user_input = input("\n👤 أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 مع السلامة!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["خروج", "exit", "quit", "q"]:
            print("👋 مع السلامة!")
            break

        # البحث عن السياق المشابه
        context = search_similar(user_input, vectordb, k=TOP_K)
        # تجهيز تاريخ المحادثة
        history_text = format_history(conversation)

        print("\n⏳ جاري التوليد...\n")
        response = generate_response(user_input, context, history_text)

        print(f"🤖 المساعد:\n{response}")
        print("-" * 60)

        # حفظ الدور في تاريخ المحادثة
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
