"""Sinotech LINE RAG: Gemini 3072 -> Pinecone Documents -> Groq."""
import base64
import hashlib
import hmac
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request
from rag_core import RagEngine, build_catalog

CONFIG = json.loads(Path(__file__).with_name("rag_config.json").read_text(encoding="utf-8"))
REQUIRED_KEYS = ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "GROQ_API_KEY", "PINECONE_API_KEY", "GEMINI_API_KEY")
VERSION = "rag-3072-v1"
logger = logging.getLogger(__name__)


def required_key(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def production_engine():
    from google import genai
    from google.genai import types
    from pinecone import Pinecone, DenseVectorQuery
    from groq import Groq

    google_client = genai.Client(api_key=required_key("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=10000))
    pinecone_client = Pinecone(api_key=required_key("PINECONE_API_KEY"), timeout=8)
    index = pinecone_client.index(CONFIG["index_name"])
    groq_client = Groq(api_key=required_key("GROQ_API_KEY"), timeout=15, max_retries=0)
    metadata, token = [], None
    while True:
        page = index.documents.fetch(namespace=CONFIG["namespace"], filter={"models": {"$exists": True}},
            include_fields=["models"], pagination_token=token)
        metadata.extend({"models": doc.get("models", [])} for doc in page.documents.values())
        token = page.pagination.next if page.pagination else None
        if not token:
            break
    catalog = build_catalog(metadata)
    if not catalog:
        raise RuntimeError("Configured Pinecone namespace has no product metadata")

    def embed(question):
        response = google_client.models.embed_content(model=CONFIG["embedding_model"],
            contents=CONFIG["query_template"].format(question=question),
            config=types.EmbedContentConfig(output_dimensionality=CONFIG["dimension"]))
        if len(response.embeddings or []) != 1:
            raise RuntimeError("Unexpected embedding count")
        vector = response.embeddings[0].values
        if len(vector) != CONFIG["dimension"] or not all(math.isfinite(v) for v in vector):
            raise RuntimeError("Unexpected embedding dimensions")
        return vector

    def search(vector, filters):
        response = index.documents.search(namespace=CONFIG["namespace"], top_k=8,
            score_by=[DenseVectorQuery(field=CONFIG["vector_field"], values=vector)], filter=filters,
            include_fields=["text", "source", "page", "models", "doc_type", "extraction", "contains_visuals"])
        return [{"score": doc.score, **{field: doc.get(field) for field in
            ("text", "source", "page", "models", "doc_type", "extraction", "contains_visuals")}} for doc in response.matches]

    def complete(system, context, question):
        response = groq_client.chat.completions.create(model="openai/gpt-oss-120b", temperature=0,
            reasoning_effort="low", include_reasoning=False, max_completion_tokens=1800,
            messages=[{"role": "system", "content": system},
                {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม:\n{question}"}])
        return response.choices[0].message.content

    return RagEngine(embed, search, complete, catalog, minimum_score=float(os.environ.get("RAG_MIN_SCORE", "0.5")))


def line_text(text, maximum=4500):
    # LINE counts text as UTF-16 units; avoid splitting an emoji surrogate pair.
    units, result = 0, []
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > maximum:
            break
        units += width
        result.append(char)
    return "".join(result)


def send_line_reply(reply_token, text):
    import httpx
    response = httpx.post("https://api.line.me/v2/bot/message/reply",
        headers={"Authorization": "Bearer " + required_key("LINE_CHANNEL_ACCESS_TOKEN")},
        json={"replyToken": reply_token, "messages": [{"type": "text", "text": line_text(text)}]}, timeout=8)
    response.raise_for_status()


def create_app(engine=None, reply=None, channel_secret=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

    @app.get("/")
    @app.get("/health")
    def health():
        missing = [name for name in REQUIRED_KEYS if not os.environ.get(name)] if engine is None else []
        return jsonify(status="ok" if not missing else "configuration_required", version=VERSION,
            index=CONFIG["index_name"], namespace=CONFIG["namespace"], dimension=CONFIG["dimension"],
            missing_environment=missing), (503 if missing else 200)

    @app.post("/webhook")
    def webhook():
        secret = channel_secret if channel_secret is not None else os.environ.get("LINE_CHANNEL_SECRET", "")
        if not secret:
            return "Webhook is not configured", 503
        body = request.get_data()
        expected = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
        provided = request.headers.get("X-Line-Signature", "")
        if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("ascii")):
            return "Invalid signature", 400
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return "Invalid JSON", 400
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            return "Invalid webhook payload", 400
        for event in payload["events"]:
            if not isinstance(event, dict):
                return "Invalid event", 400
            message = event.get("message")
            if event.get("type") != "message" or not isinstance(message, dict) or message.get("type") != "text":
                continue
            question = message.get("text")
            reply_token = event.get("replyToken")
            if not isinstance(question, str) or not reply_token:
                continue
            try:
                current_engine = engine if engine is not None else production_engine()
                answer = current_engine.answer(question)
            except Exception as exc:
                logger.error("RAG request failed (%s)", type(exc).__name__)
                answer = "ระบบค้นข้อมูลขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้งหรือติดต่อแอดมินครับ"
            try:
                (reply or send_line_reply)(reply_token, line_text(answer))
            except Exception as exc:
                logger.error("LINE reply failed (%s)", type(exc).__name__)
                return "Reply failed", 502
        return "OK", 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
