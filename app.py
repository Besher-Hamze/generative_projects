import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests
import json

# إعدادات الصفحة
st.set_page_config(
    page_title="مولد مشاريع التخرج",
    page_icon="🎓",
    layout="wide"
)

# دالة للاتصال مع Ollama
def call_ollama(prompt, model="qwen2.5:0.5b"):
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

# تحميل وتجهيز البيانات
@st.cache_resource
def initialize_vectordb():
    """إنشاء قاعدة البيانات المتجهة"""
    
    # قراءة ملف المشاريع
    if not os.path.exists('projects.txt'):
        st.error("❌ ملف projects.txt غير موجود!")
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
    
    # إنشاء VectorDB
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectordb

# البحث عن مشاريع مشابهة
def search_similar(query, vectordb, k=5):
    """البحث عن مشاريع مشابهة"""
    results = vectordb.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in results])
    return context

# توليد المشروع
def generate_project(user_input, context, conversation_history=""):
    """توليد مشروع جديد مع محادثة"""
    
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

# واجهة التطبيق
def main():
    
    # العنوان
    st.title("🎓 مولد أفكار مشاريع التخرج الذكي")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 👋 مرحباً!
        أنا مساعدك الذكي لتوليد أفكار مشاريع التخرج المبتكرة.
        
        **يمكنني مساعدتك في:**
        - توليد أفكار مشاريع جديدة بناءً على اهتماماتك
        - اقتراح التقنيات والأدوات المناسبة
        - الإجابة على أسئلتك حول المشاريع
        - تقديم خطوات التنفيذ التفصيلية
        """)
    
    with col2:
        st.info("""
        **💡 أمثلة للأسئلة:**
        - اقترح مشروع في مجال الصحة
        - أريد مشروع يستخدم التعلم العميق
        - ما التقنيات المناسبة لمشروع روبوت؟
        """)
    
    st.markdown("---")
    
    # تحميل قاعدة البيانات
    with st.spinner("⏳ جاري تحميل قاعدة البيانات..."):
        vectordb = initialize_vectordb()
    
    if vectordb is None:
        st.stop()
    
    st.success("✅ النظام جاهز!")
    
    # تهيئة سجل المحادثة
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # عرض المحادثات السابقة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # صندوق الإدخال
    if prompt := st.chat_input("💬 اكتب طلبك هنا... (مثال: أريد مشروع في مجال الذكاء الاصطناعي)"):
        
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # توليد الرد
        with st.chat_message("assistant"):
            with st.spinner("🤔 جاري التفكير وتوليد الإجابة..."):
                
                # البحث عن مشاريع مشابهة
                context = search_similar(prompt, vectordb, k=5)
                
                # بناء سياق المحادثة
                conversation_history = ""
                if len(st.session_state.messages) > 1:
                    recent_messages = st.session_state.messages[-6:-1]
                    conversation_history = "المحادثة السابقة:\n"
                    for msg in recent_messages:
                        role = "الطالب" if msg["role"] == "user" else "المساعد"
                        conversation_history += f"{role}: {msg['content']}\n"
                
                # توليد الرد
                response = generate_project(prompt, context, conversation_history)
                
                st.markdown(response)
        
        # حفظ الرد
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # زر لمسح المحادثة
    st.sidebar.title("⚙️ الإعدادات")
    if st.sidebar.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()
    
    # معلومات
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 الإحصائيات
    - **عدد المشاريع المحفوظة:** محمّلة ✅
    - **النموذج المستخدم:** Qwen 2.5 (7B)
    - **قاعدة البيانات:** ChromaDB
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ معلومات
    تم تطوير هذا النظام باستخدام:
    - **Ollama** (نموذج محلي)
    - **LangChain** (إطار العمل)
    - **ChromaDB** (قاعدة البيانات)
    - **Streamlit** (الواجهة)
    """)

if __name__ == "__main__":
    main()