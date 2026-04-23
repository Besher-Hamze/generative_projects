import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import hashlib
from pymongo import MongoClient
from typing import List

# إعدادات التشفير
def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

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

# إعداد MongoDB
try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = mongo_client["grad_projects_db"]
    users_collection = db["users"]
    answers_collection = db["answers"]
    questions_collection = db["questions"]
    # فحص الاتصال
    mongo_client.server_info()
    print("✅ تم الاتصال بقاعدة بيانات MongoDB بنجاح!")
    
    # زراعة الأسئلة الافتراضية إذا لم تكن موجودة
    if questions_collection.count_documents({}) == 0:
        default_questions = [
            {"id": 1, "text": "ما هي لغات البرمجة التي تتقنها أو تفضل العمل بها؟ (مثل: Python, Java, JavaScript)"},
            {"id": 2, "text": "هل تفضل تطوير واجهات المستخدم (Frontend) أم تطوير الأنظمة الخلفية (Backend) أم كلاهما؟"},
            {"id": 3, "text": "ما هي مجالات اهتمامك الرئيسية في التقنية؟ (مثل: الذكاء الاصطناعي، تطبيقات الموبايل، أمن المعلومات)"},
            {"id": 4, "text": "هل تميل للمشاريع البحثية النظرية والخوارزميات، أم المشاريع التطبيقية والعملية؟"}
        ]
        questions_collection.insert_many(default_questions)
        print("✅ تم إضافة الأسئلة الافتراضية بنجاح!")
        
except Exception as e:
    print(f"⚠️ تحذير: تعذر الاتصال بـ MongoDB. يرجى التأكد من تشغيل الخدمة. الخطأ: {e}")

# نماذج البيانات (Pydantic Models)
class ChatRequest(BaseModel):
    query: str
    conversation_history: str = ""
    model: str = "qwen2.5:7b"

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AnswerSubmit(BaseModel):
    email: str
    answers: List[str]

class RecommendRequest(BaseModel):
    email: str
    query: str = "اقترح لي مشروع تخرج مناسب بناءً على مهاراتي واهتماماتي"
    model: str = "qwen2.5:7b"

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

def generate_project(user_input, context, conversation_history="", model="qwen2.5:7b"):
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

    return call_ollama(prompt, model=model)

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """الاندبوينت الخاصة بالأسئلة والمحادثة ليتم استدعاؤها من Flutter"""
    if vectordb is None:
        raise HTTPException(status_code=500, detail="قاعدة البيانات غير مهيأة بعد.")
    
    # البحث عن السياق المشابه
    context = search_similar(request.query, vectordb, k=5)
    
    # توليد الرد من Ollama
    response = generate_project(request.query, context, request.conversation_history, request.model)
    
    return {"message": response}

@app.post("/api/register")
def register(user: UserRegister):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="البريد الإلكتروني مسجل مسبقاً")
    
    user_dict = user.model_dump() if hasattr(user, 'model_dump') else user.dict()
    user_dict['password'] = get_password_hash(user.password)
    users_collection.insert_one(user_dict)
    return {
        "message": "تم التسجيل بنجاح",
        "name": user.name,
        "email": user.email,
        "isAnswered": False
    }

@app.post("/api/login")
def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or db_user['password'] != get_password_hash(user.password):
        raise HTTPException(status_code=401, detail="البريد الإلكتروني أو كلمة المرور غير صحيحة")
    
    has_answers = answers_collection.find_one({"email": user.email}) is not None
    
    return {
        "message": "تم تسجيل الدخول بنجاح", 
        "name": db_user.get('name', ''), 
        "email": db_user.get('email', ''),
        "isAnswered": has_answers
    }

@app.get("/api/questions")
def get_questions():
    questions = list(questions_collection.find({}, {"_id": 0}))
    return {"questions": questions}

@app.post("/api/answers")
def submit_answers(data: AnswerSubmit):
    if not users_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    
    answers_collection.update_one(
        {"email": data.email},
        {"$set": {"answers": data.answers}},
        upsert=True
    )
    return {"message": "تم حفظ الإجابات بنجاح"}

@app.post("/api/recommend")
def recommend_project(request: RecommendRequest):
    if vectordb is None:
        raise HTTPException(status_code=500, detail="قاعدة البيانات المتجهة غير مهيأة بعد.")
    
    user_answers = answers_collection.find_one({"email": request.email})
    behavior_context = ""
    if user_answers and "answers" in user_answers:
        behavior_context = "\nإجابات الطالب السابقة على أسئلة حول اهتماماته ومهاراته والمواد التي يفضلها:\n- " + "\n- ".join(user_answers["answers"])
    
    context = search_similar(request.query, vectordb, k=5)
    
    prompt = f"""أنت مساعد ذكي متخصص في توليد أفكار مشاريع تخرج في هندسة المعلوماتية والذكاء الاصطناعي.

المشاريع المشابهة الموجودة في قاعدة البيانات للحصول على إلهام:
{context}

{behavior_context}

سؤال/طلب الطالب الحالي: {request.query}

المطلوب منك:
- بناءً على إجابات الطالب السابقة في الأعلى لتحديد شخصيته واهتماماته (مهم جداً)، قم باقتراح فكرة مشروع تخرج إبداعية ومناسبة جداً لمستواه ورغباته.
- يجب أن يغطي المقترح:
  1. عنوان المشروع
  2. وصف تفصيلي للمشروع يوضح سبب اختيارك لهذه الفكرة للطالب وارتباطها المباشر مع إجاباته السابقة
  3. التقنيات والأدوات المقترحة التي تتوافق مع قدرات الطالب
  4. المكتبات المطلوبة للبدء
  5. خطوات التنفيذ الأساسية
  6. الفوائد المتوقعة

تأكد أن:
✓ المشروع مبتكر ومخصص خصيصاً لوضع الطالب
✓ التقنيات مناسبة وحديثة
✓ الشرح واضح ومنظم
✓ الإجابة بالعربية بشكل كامل

الرد:"""

    response = call_ollama(prompt, model=request.model)
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    # تشغيل التطبيق في بيئة الإنتاج
    uvicorn.run("app:app", host="0.0.0.0", port=8000)