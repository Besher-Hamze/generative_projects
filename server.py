"""
خادم FastAPI يستخدم منطق chat.py ويقدّم واجهة HTML على نفس المنفذ.

التشغيل:
    pip install fastapi uvicorn
    python server.py
    ثم افتح المتصفح على:  http://localhost:8000
"""

from typing import List
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# نستورد المنطق من chat.py مباشرة (لا نكرر الكود)
from chat import (
    initialize_vectordb,
    search_similar,
    generate_response,
    format_history,
    DEFAULT_MODEL,
    TOP_K,
)

# نقطة Ollama لجلب قائمة النماذج المثبتة
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

app = FastAPI(title="مولد مشاريع التخرج")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغير عالمي لقاعدة البيانات المتجهة
vectordb = None


# ==================== نماذج البيانات ====================
class Message(BaseModel):
    role: str      # "user" أو "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    model: str = DEFAULT_MODEL


# ==================== أحداث دورة الحياة ====================
@app.on_event("startup")
def startup_event():
    global vectordb
    print("⏳ جاري تحميل قاعدة البيانات...")
    vectordb = initialize_vectordb()
    if vectordb:
        print("✅ الخادم جاهز على http://localhost:8000")
    else:
        print("⚠️ فشل تحميل قاعدة البيانات.")


# ==================== المسارات (Routes) ====================
@app.get("/")
def index():
    """تقديم صفحة الواجهة الرسومية."""
    return FileResponse("index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    """نقطة الاتصال الرئيسية للمحادثة."""
    if vectordb is None:
        raise HTTPException(
            status_code=500,
            detail="قاعدة البيانات المتجهة غير مهيأة بعد."
        )

    # البحث عن أقرب مشاريع
    context = search_similar(request.query, vectordb, k=TOP_K)

    # تجهيز تاريخ المحادثة (نحوّل Pydantic إلى dict)
    history_dicts = [m.model_dump() for m in request.history]
    history_text = format_history(history_dicts)

    # توليد الرد
    response = generate_response(
        request.query,
        context,
        history_text,
        request.model,
    )

    return {"message": response}


@app.get("/api/health")
def health():
    return {"status": "ok", "db_loaded": vectordb is not None}


@app.get("/api/models")
def get_models():
    """جلب قائمة النماذج المثبتة في Ollama محلياً."""
    try:
        res = requests.get(OLLAMA_TAGS_URL, timeout=5)
        res.raise_for_status()
        data = res.json()
        # كل عنصر في models يحوي 'name' مثل 'qwen2.5:7b'
        models = sorted([m["name"] for m in data.get("models", [])])
        if not models:
            models = [DEFAULT_MODEL]
        # اختيار النموذج الافتراضي: إذا كان DEFAULT_MODEL موجوداً نستخدمه،
        # وإلا نستخدم أول نموذج في القائمة
        default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
        return {"models": models, "default": default}
    except Exception as e:
        # رجوع احتياطي: نرجع النموذج الافتراضي فقط مع رسالة الخطأ
        return {
            "models": [DEFAULT_MODEL],
            "default": DEFAULT_MODEL,
            "error": f"تعذر الاتصال بـ Ollama: {str(e)}",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)