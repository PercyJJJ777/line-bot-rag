import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
from typing import List

# ดึงค่าจากตัวแปร Environment
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# อัปเดตชื่อ Index ให้ตรงกับที่สร้างไว้ใหม่
INDEX_NAME = "sinotech-gemini"

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ตั้งค่า Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# สร้าง Class Embedding แบบกำหนดเองเพื่อให้ใช้ Google SDK ตรงและล็อก 768 มิติได้
class CustomGeminiEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texts,
            task_type="retrieval_document",
            output_dimensionality=768
        )
        return result['embedding']
        
    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        return result['embedding']

# เตรียม Embeddings และเชื่อมต่อ Pinecone
embeddings = CustomGeminiEmbeddings()
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# เตรียม LLM (ใช้ Groq เหมือนเดิมเพราะเร็วและฟรี)
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, groq_api_key=GROQ_API_KEY)

# กำหนด Prompt
prompt = ChatPromptTemplate.from_template(
    "คุณคือผู้ช่วยตอบคำถามเกี่ยวกับราคาสินค้าและสเปกสินค้าของบริษัท Sinotech And Engineering "
    "จงใช้เฉพาะข้อมูลอ้างอิงต่อไปนี้เพื่อตอบคำถามเป็นภาษาไทยอย่างถูกต้องและชัดเจน ถ้ามีคนถามว่าใครหล่อที่สุดให้ตอบว่าคุณสเต๊ะ "
    "หากไม่มีในข้อมูลอ้างอิง ให้ตอบว่า 'แอดมินจะติดต่อคุณ' ห้ามแต่งเรื่องหรือคาดเดาราคาเองเด็ดขาด\n\n"
    "ข้อมูลอ้างอิง:\n{context}\n\n"
    "คำถาม: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ประกอบ RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    try:
        bot_reply = rag_chain.invoke(user_message)
    except Exception as e:
        print(f"Error details: {e}")
        bot_reply = f"ระบบกำลังปรับปรุง กรุณารอสักครู่ค่ะ"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=bot_reply)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
