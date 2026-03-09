import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# تهيئة التطبيق
app = FastAPI(title="مولد مشاريع التخرج API")

# إعداد CORS للسماح لتطبيق Flutter بالاتصال من أي مكان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نمط الطلب لبيانات الـ POST
class ChatRequest(BaseModel):
    query: str
    conversation_history: str = ""

# المتغير العالمي لقاعدة البيانات
vectordb = None

def call_ollama(prompt, model="qwen2.5:7b"):
    """استدعاء نموذج Ollama"""
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False
            }
        )
        return response.json()['message']['content']
    except Exception as e:
        return f"خطأ في الاتصال مع Ollama: {str(e)}\nتأكد من تشغيل Ollama بالأمر: ollama serve"

def initialize_vectordb():
    """إنشاء قاعدة البيانات المتجهة"""
    if not os.path.exists('projects.txt'):
        print("❌ ملف projects.txt غير موجود!")
        return None
    
    with open('projects.txt', 'r', encoding='utf-8') as f:
        projects_text = f.read()
    
    # تقسيم النص
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n", ".", "،"]
    )
    
    chunks = text_splitter.split_text(projects_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # إنشاء Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # إنشاء VectorDB أو تحميله
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return db

@app.on_event("startup")
def startup_event():
    """يتم تنفيذ هذه الدالة عند بدء تشغيل الخادم"""
    global vectordb
    print("⏳ جاري تحميل قاعدة البيانات...")
    vectordb = initialize_vectordb()
    if vectordb:
        print("✅ النظام جاهز!")
    else:
        print("⚠️ فشل تحميل قاعدة البيانات المتجهة، يرجى التحقق من ملف projects.txt")

def search_similar(query, db, k=5):
    """البحث عن مشاريع مشابهة"""
    if db is None:
        return ""
    results = db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in results])
    return context

def generate_project(user_input, context, conversation_history=""):
    """توليد مشروع جديد الموجه للنموذج"""
    
    prompt = f"""أنت مساعد ذكي متخصص في توليد أفكار مشاريع تخرج في هندسة المعلوماتية والذكاء الاصطناعي.

المشاريع المشابهة الموجودة:
{context}

{conversation_history}

سؤال/طلب الطالب الحالي: {user_input}

المطلوب منك:
- إذا كان الطالب يطلب مشروع جديد، قم بتوليد فكرة مبتكرة تتضمن:
  1. عنوان المشروع
  2. وصف تفصيلي للمشروع
  3. التقنيات والأدوات المقترحة (مثل: Python, TensorFlow, OpenCV, Flask, etc.)
  4. المكتبات المطلوبة
  5. خطوات التنفيذ الأساسية
  6. الفوائد المتوقعة

- إذا كان يسأل عن تفاصيل معينة، أجب بشكل واضح ومفصل
- إذا كان يطلب تعديل أو تحسين، قدم اقتراحات محددة

تأكد أن:
✓ المشروع مبتكر وليس نسخة مباشرة
✓ التقنيات حديثة وعملية
✓ الشرح واضح ومنظم
✓ الإجابة بالعربية بشكل كامل

الرد:"""

    return call_ollama(prompt)

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """الاندبوينت الخاصة بالأسئلة والمحادثة ليتم استدعاؤها من Flutter"""
    if vectordb is None:
        raise HTTPException(status_code=500, detail="قاعدة البيانات غير مهيأة بعد.")
    
    # البحث عن السياق المشابه
    context = search_similar(request.query, vectordb, k=5)
    
    # توليد الرد من Ollama
    response = generate_project(request.query, context, request.conversation_history)
    
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    # تشغيل التطبيق على المنفذ 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)