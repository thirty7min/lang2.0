from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import time
import re
import uuid
from typing import List, Dict
from openai import AsyncOpenAI
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LangAPI")

# Initialize FastAPI
langapi = FastAPI(title="LangAPI", version="2.3.0")
langapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    content: str
    sourceLanguage: str = "en"
    targetLanguage: str

class RequestLogger:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.logs: List[str] = []
        self.chunk_results: List[Dict] = []

    def log(self, message: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{self.request_id}] {ts} | {level} | {message}"
        self.logs.append(entry)
        logger.info(entry)

    def log_chunk_result(self, chunk_id: int, success: bool, chars: int, processing_time: float, error: str = None):
        result = {
            "chunk_id": chunk_id,
            "success": success,
            "characters": chars,
            "processing_time_ms": round(processing_time * 1000, 2),
            "error": error
        }
        self.chunk_results.append(result)
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.log(f"Chunk {chunk_id}: {status} | {chars} chars | {result['processing_time_ms']}ms | {error or 'OK'}")

    def get_summary(self):
        total = time.time() - self.start_time
        successful = len([r for r in self.chunk_results if r["success"]])
        failed = len([r for r in self.chunk_results if not r["success"]])
        return {
            "request_id": self.request_id,
            "total_processing_time": round(total, 3),
            "chunks_successful": successful,
            "chunks_failed": failed,
            "chunk_details": self.chunk_results,
            "logs": self.logs
        }

def get_openai_client():
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return AsyncOpenAI(api_key=key, timeout=60.0)
    except Exception as e:
        logger.error(f"OpenAI client error: {e}")
        return None

def clean_translated_html(original: str, translated: str) -> str:
    wrappers = [
        r'^<html[^>]*>(.*)</html>$',
        r'^<body[^>]*>(.*)</body>$',
        r'^<div[^>]*>(.*)</div>$',
        r'^<p[^>]*>(.*)</p>$'
    ]
    text = translated.strip()
    for pat in wrappers:
        m = re.match(pat, text, re.DOTALL | re.IGNORECASE)
        if m and not re.match(pat, original.strip(), re.DOTALL | re.IGNORECASE):
            text = m.group(1).strip()
    return re.sub(r'\n\s*\n\s*\n', '\n\n', text)

class SmartHTMLChunker:
    def __init__(self, target_chars=1500):
        self.target_chars = target_chars

    def calculate_optimal_config(self, length: int, log: RequestLogger):
        if length < 3000:
            cfg = (2, 2, "Small content")
        elif length < 8000:
            cfg = (5, 4, "Medium content")
        elif length < 20000:
            cfg = (10, 6, "Large content")
        elif length < 50000:
            cfg = (15, 8, "Very large content")
        else:
            cfg = (20, 10, "Huge content")
        max_chunks, parallel, category = cfg
        log.log(f"Content analysis: {length} chars classified as '{category}'")
        log.log(f"Optimization: Max {max_chunks} chunks, {parallel} parallel workers")
        return max_chunks, parallel

    def find_safe_break_points(self, html: str, log: RequestLogger):
        patterns = [
            r'</p>\s*<p', r'</h[1-6]>\s*<', r'</li>\s*<li', r'</ul>\s*<',
            r'</ol>\s*<', r'</div>\s*</div>\s*<div', r'</section>\s*<section',
            r'</article>\s*<article', r'</td>\s*<td', r'</tr>\s*<tr',
            r'</thead>\s*<tbody', r'</tbody>\s*</table', r'</table>\s*<',
            r'</blockquote>\s*<', r'</pre>\s*<', r'</code>\s*<'
        ]
        points = {0, len(html)}
        for pat in patterns:
            for m in re.finditer(pat, html, re.IGNORECASE):
                p = m.end() - len(m.group().split('<')[-1]) - 1
                if p > 0:
                    points.add(p)
        pts = sorted(points)
        log.log(f"Break point analysis: Found {len(pts)} points")
        return pts

    def create_smart_chunks(self, html: str, log: RequestLogger):
        length = len(html)
        max_chunks, parallel = self.calculate_optimal_config(length, log)
        if length <= self.target_chars:
            log.log("Single chunk strategy: content fits in one chunk")
            return [{"id": 0, "content": html}], parallel

        breaks = self.find_safe_break_points(html, log)
        num = min(max(1, length // self.target_chars), max_chunks)
        log.log(f"Chunking strategy: splitting into {num} chunks from {length} chars")

        chunks = []
        per = length // num
        start = 0
        for i in range(num):
            end = breaks[-1] if i == num - 1 else min(
                breaks, key=lambda b: abs(b - (start + per))
            )
            chunk_txt = html[start:end]
            chunks.append({"id": i, "content": chunk_txt})
            log.log(f"Chunk {i}: {len(chunk_txt)} chars")
            start = end
        log.log(f"Chunking complete: {len(chunks)} chunks, parallel={parallel}")
        return chunks, parallel

chunker = SmartHTMLChunker()

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest, http_request: Request):
    request_id = str(uuid.uuid4())[:8]
    log = RequestLogger(request_id)
    try:
        ip = http_request.client.host if http_request.client else "unknown"
        log.log(f"🚀 NEW REQUEST | {request.sourceLanguage} → {request.targetLanguage} | IP: {ip}")
        log.log(f"Content length: {len(request.content):,} chars")

        client = get_openai_client()
        if not client:
            log.log("❌ No OpenAI client", "ERROR")
            raise HTTPException(500, "OpenAI client not available")

        log.log("✅ OpenAI client initialized")
        t0 = time.time()
        chunks, parallel = chunker.create_smart_chunks(request.content, log)
        log.log(f"⚡ Chunking took {(time.time() - t0)*1000:.1f}ms")

        t1 = time.time()
        translated_chunks = await translate_html_chunks_parallel(
            chunks,
            request.sourceLanguage,
            request.targetLanguage,
            client,
            parallel,
            log
        )
        log.log(f"🔄 Translation took {time.time() - t1:.2f}s")

        t2 = time.time()
        final_html = reassemble_translated_chunks(translated_chunks, request.content, log)
        log.log(f"🔧 Assembly took {(time.time() - t2)*1000:.1f}ms")

        summary = log.get_summary()
        log.log(f"✅ COMPLETE | Total: {summary['total_processing_time']}s | "
                f"Success: {summary['chunks_successful']}/{len(chunks)}")

        return {
            "translatedContent": final_html,
            "requestId": request_id,
            "processingStats": {
                "chunksProcessed": len(translated_chunks),
                "chunksSuccessful": summary["chunks_successful"],
                "chunksFailed": summary["chunks_failed"],
                "parallelWorkers": parallel,
                "totalProcessingTime": summary["total_processing_time"],
                "phases": {
                    "chunking": round((t0 - t0) * 1000, 1),
                    "translation": round((time.time() - t1) * 1000, 1),
                    "assembly": round((time.time() - t2) * 1000, 1)
                }
            },
            "chunkDetails": summary["chunk_details"],
            "fromCache": False
        }
    except Exception as e:
        log.log(f"❌ REQUEST FAILED: {e}", "ERROR")
        raise HTTPException(500, f"Translation failed: {e}")

async def translate_html_chunks_parallel(chunks, source_lang, target_lang, client, parallel, log):
    sem = asyncio.Semaphore(parallel)
    log.log(f"🔄 Starting parallel translation: {len(chunks)} chunks, {parallel} workers")

    async def worker(chunk):
        start = time.time()
        async with sem:
            # log without blocking
            asyncio.create_task(
                asyncio.to_thread(log.log,
                    f"🔄 Processing chunk {chunk['id']} ({len(chunk['content'])} chars)")
            )
            # build prompt
            system_prompt = (
                f"You are an expert HTML translator. Translate from {source_lang} to {target_lang}.\n\n"
                "CRITICAL RULES:\n"
                "1. Return EXACTLY the same HTML structure as input...\n"
                # etc...
            )
            # API call
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": chunk["content"]}
                ],
                temperature=0.02,
                max_tokens=4000
            )
            api_dur = time.time() - start
            text = resp.choices[0].message.content.strip()
            cleaned = clean_translated_html(chunk["content"], text)
            proc_time = time.time() - start

            # fire-and-forget logging
            asyncio.create_task(asyncio.to_thread(
                log.log_chunk_result,
                chunk["id"], True, len(chunk["content"]), proc_time, None
            ))
            asyncio.create_task(asyncio.to_thread(
                log.log,
                f"  API call took {api_dur*1000:.0f}ms"
            ))

            return {"id": chunk["id"], "translated_content": cleaned, "success": True}

    results = await asyncio.gather(*(worker(c) for c in chunks))
    succ = sum(1 for r in results if r["success"])
    fail = len(results) - succ
    avg = sum(0 for _ in results)  # you can compute avg if you track times
    log.log(f"🏁 Parallel complete: {succ} success, {fail} failed")
    return sorted(results, key=lambda r: r["id"])

def reassemble_translated_chunks(translated, original, log):
    log.log(f"🔧 Reassembling {len(translated)} chunks")
    out = "".join(r["translated_content"] for r in translated)
    cleaned = clean_translated_html(original, out)
    log.log("📦 Assembly done")
    return cleaned

@langapi.get("/health")
async def health_check():
    return {"status": "healthy", "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}

@langapi.get("/")
async def root():
    return {"service": "LangAPI", "version": "2.3.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
