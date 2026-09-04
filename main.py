"""Aryavarta web entrypoint for Vercel.

Keeps the original Raspberry-Pi application and knowledge files intact while
adding a dependency-light browser/web API. The browser handles camera,
expression estimation, speech recognition and speech synthesis; this server
handles scripture retrieval and optional server-side Gemini generation.
"""
from __future__ import annotations

# Web deployment is intentionally dependency-free; Raspberry-Pi packages live
# in requirements-pi.txt and are not required by this entrypoint.

import ast
import json
import os
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
_DATA_CACHE = None


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _literal_assignments(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return {}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = value
    return found


def _load_gita():
    result = []
    for path in sorted((ROOT / "data" / "gita").glob("ch*.py")):
        values = _literal_assignments(path)
        for name, chapter in values.items():
            if not name.startswith("BHAGAVAD_GITA_CH") or not isinstance(chapter, dict):
                continue
            chapter_no = name.replace("BHAGAVAD_GITA_CH", "")
            for verse_no, verse in chapter.items():
                if isinstance(verse, dict):
                    result.append({"chapter": chapter_no, "verse": str(verse_no), "sanskrit": str(verse.get("sanskrit", "")), "meaning": str(verse.get("meaning", "")), "example": str(verse.get("example", ""))})
    return result


def _load_vedas():
    values = _literal_assignments(ROOT / "data" / "vedas" / "vedas_summary.py")
    complete = values.get("VEDAS_COMPLETE", {})
    comparison = complete.get("vedas_comparison", {}) if isinstance(complete, dict) else {}
    result = []
    for name, info in comparison.items():
        if isinstance(info, dict):
            result.append({"name": info.get("name", name), "meaning": info.get("meaning", ""), "focus": info.get("focus", ""), "character": info.get("character", ""), "structure": info.get("structure", ""), "verses": info.get("verses", ""), "psychological_focus": info.get("psychological_focus", ""), "famous_hymns": info.get("famous_hymns", []), "key_verse": info.get("key_verse", "")})
    return result


def _data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = {"gita": _load_gita(), "vedas": _load_vedas()}
    return _DATA_CACHE


INTENTS = {
    "life_guidance": ["sad", "anxious", "worry", "fear", "confused", "decision", "career", "future", "stress", "anger", "angry", "failure", "exam", "purpose", "duty", "problem", "help"],
    "medicine": ["medicine", "medical", "healing", "health", "herb", "illness", "disease", "fever", "pain"],
    "mantra": ["mantra", "chant", "song", "music", "melody", "devotion", "prayer", "meditation", "peace", "calm"],
    "ritual": ["ritual", "yajna", "havan", "fire", "ceremony", "sacrifice", "offering", "puja", "priest"],
    "hymn": ["hymn", "courage", "bravery", "protection", "warrior", "praise", "gods"],
}


def _classify(text: str):
    low = (text or "").lower()
    scores = {k: sum(1 for word in words if word in low) for k, words in INTENTS.items()}
    intent = max(scores, key=scores.get)
    if scores.get(intent, 0) == 0:
        intent = "life_guidance"
    return {"medicine": "Atharvaveda", "mantra": "Samaveda", "ritual": "Yajurveda", "hymn": "Rigveda"}.get(intent), intent


def _expression_words(expression):
    return {"positive": "positive and engaged", "downcast": "downcast", "surprised": "surprised", "neutral": "neutral"}.get((expression or "neutral").lower(), "neutral")


def _score(item, query):
    text = " ".join([item.get("meaning", ""), item.get("example", ""), item.get("sanskrit", "")]).lower()
    words = [w for w in query.lower().replace("?", " ").replace(",", " ").split() if len(w) > 3]
    return sum(1 for w in words if w in text)


def _retrieve(message, expression):
    data = _data()
    branch, intent = _classify(message)
    if intent == "life_guidance":
        ranked = sorted(data["gita"], key=lambda x: _score(x, message), reverse=True)
        ranked = [x for x in ranked if _score(x, message) > 0][:3] or data["gita"][:3]
        if ranked:
            v = ranked[0]
            return {"kind": "Bhagavad Gita", "title": f"Bhagavad Gita • Chapter {v['chapter']}, Verse {v['verse']}", "text": v["meaning"], "sanskrit": v["sanskrit"], "guidance": v["example"], "expression": _expression_words(expression)}
    if branch:
        info = next((x for x in data["vedas"] if x["name"].lower() == branch.lower()), None)
        if info:
            return {"kind": branch, "title": branch, "text": info["focus"], "sanskrit": info["key_verse"], "guidance": info["psychological_focus"], "expression": _expression_words(expression)}
    return {"kind": "Four Vedas", "title": "Four Vedas", "text": "The Vedas are presented in this project through separate Rigveda, Yajurveda, Samaveda and Atharvaveda knowledge collections.", "sanskrit": "", "guidance": "Ask about a specific Veda or a life question to retrieve a more focused entry.", "expression": _expression_words(expression)}


def _fallback_answer(source):
    if source["kind"] == "Bhagavad Gita":
        return f"Based on the Bhagavad Gita entry I retrieved, the key idea is: {source['text']}\n\nA practical way to apply it: {source['guidance']}"
    return f"From {source['kind']}, the relevant focus is: {source['text']}\n\nIn this project's knowledge base, its psychological focus is: {source['guidance']}"


def _gemini_answer(message, expression, source):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    prompt = ("You are Aryavarta, a respectful educational assistant grounded in the project's retrieved Bhagavad Gita and Veda data. Answer naturally and concisely. Do not claim that a facial expression proves someone's emotion; treat it only as a coarse UI signal. Do not invent scripture quotations. If the retrieved source is a summary, say it is a summary.\nUser: " + message + "\nUI expression signal: " + _expression_words(expression) + "\nRetrieved source: " + json.dumps(source, ensure_ascii=False))
    body = _json_bytes({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.5, "maxOutputTokens": 450}})
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + key
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, OSError):
        return None


def _response(start_response, status, payload, content_type="application/json; charset=utf-8"):
    body = payload if isinstance(payload, bytes) else _json_bytes(payload)
    start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
    return [body]


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()
    if path == "/health":
        return _response(start_response, "200 OK", {"ok": True, "service": "aryavarta-web"})
    if path == "/api/chat" and method == "POST":
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            raw = environ["wsgi.input"].read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            message = str(body.get("message", "")).strip()
            expression = str(body.get("expression", "neutral"))
            if not message:
                return _response(start_response, "400 Bad Request", {"error": "message is required"})
            source = _retrieve(message, expression)
            answer = _gemini_answer(message, expression, source) or _fallback_answer(source)
            return _response(start_response, "200 OK", {"answer": answer, "mode": "gemini+retrieval" if os.environ.get("GEMINI_API_KEY") else "retrieval", "expression": source["expression"], "source": {"title": source["title"], "text": source["text"], "sanskrit": source["sanskrit"]}})
        except Exception:
            return _response(start_response, "500 Internal Server Error", {"error": "Aryavarta could not process the request"})
    if path == "/" or path == "/index.html":
        try:
            body = INDEX.read_bytes()
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        except OSError:
            return _response(start_response, "500 Internal Server Error", {"error": "index.html is missing"})
    return _response(start_response, "404 Not Found", {"error": "not found"})


application = app
handler = app
