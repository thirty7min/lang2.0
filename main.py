from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import time
import re
import uuid
import json
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
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

# Initialize LangAPI
langapi = FastAPI(title="LangAPI", version="3.0.0")

# Enable CORS
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
        self.logs = []
        self.chunk_results = []
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{self.request_id}] {timestamp} | {level} | {message}"
        self.logs.append(log_entry)
        logger.info(log_entry)
    
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
        self.log(f"Chunk {chunk_id}: {status} | {chars} chars | {result['processing_time_ms']}ms")
    
    def get_summary(self):
        total_time = time.time() - self.start_time
        successful_chunks = len([r for r in self.chunk_results if r['success']])
        failed_chunks = len([r for r in self.chunk_results if not r['success']])
        
        return {
            "request_id": self.request_id,
            "total_processing_time": round(total_time, 3),
            "chunks_successful": successful_chunks,
            "chunks_failed": failed_chunks,
            "chunk_details": self.chunk_results,
            "logs": self.logs
        }

# Global OpenAI client - initialize once
_openai_client = None

def get_openai_client():
    """Get OpenAI client - initialize once and reuse"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        _openai_client = AsyncOpenAI(
            api_key=api_key, 
            timeout=30.0,  # Reduced timeout
            max_retries=2   # Reduced retries
        )
    return _openai_client

def clean_translated_html(original_content, translated_content):
    """Lightweight HTML cleaning"""
    cleaned = translated_content.strip()
    
    # Only remove obvious wrapper tags that weren't in original
    if not original_content.strip().startswith('<html') and cleaned.startswith('<html'):
        cleaned = re.sub(r'^<html[^>]*>(.*)</html>$', r'\1', cleaned, flags=re.DOTALL)
    if not original_content.strip().startswith('<body') and cleaned.startswith('<body'):
        cleaned = re.sub(r'^<body[^>]*>(.*)</body>$', r'\1', cleaned, flags=re.DOTALL)
    
    return cleaned.strip()

class FastHTMLChunker:
    def __init__(self):
        # Larger chunks = fewer API calls = faster processing
        self.target_chars = 4000  # Increased from 1500
        self.max_chunks = 15      # Reduced max chunks
        
    def calculate_optimal_config(self, content_length, req_logger):
        """Aggressive optimization for speed"""
        
        if content_length < 5000:
            config = (1, 5, "Small - no chunking needed")
        elif content_length < 15000:
            config = (3, 15, "Medium content")  # Increased parallel limit
        elif content_length < 40000:
            config = (8, 25, "Large content")   # Much higher parallel
        else:
            config = (15, 30, "Very large content")  # Max parallel
        
        max_chunks, parallel_limit, size_category = config
        req_logger.log(f"Fast mode: {content_length} chars → {size_category}")
        req_logger.log(f"Config: Max {max_chunks} chunks, {parallel_limit} parallel")
        
        return max_chunks, parallel_limit
    
    def find_fast_break_points(self, html_content):
        """Fast break point detection - only essential patterns"""
        break_points = [0]
        
        # Only the most reliable break patterns for speed
        essential_patterns = [
            r'</p>\s*<p',
            r'</div>\s*<div',
            r'</h[1-6]>\s*<',
            r'</li>\s*<li',
            r'</tr>\s*<tr'
        ]
        
        for pattern in essential_patterns:
            for match in re.finditer(pattern, html_content, re.IGNORECASE):
                break_point = match.end() - len(match.group().split('<')[-1]) - 1
                if break_point > 0:
                    break_points.append(break_point)
        
        break_points.append(len(html_content))
        return sorted(list(set(break_points)))
    
    def create_fast_chunks(self, html_content, req_logger):
        """Create larger chunks for faster processing"""
        
        content_length = len(html_content)
        max_chunks, parallel_limit = self.calculate_optimal_config(content_length, req_logger)
        
        # Skip chunking for small content
        if content_length <= self.target_chars:
            req_logger.log("Single chunk: Content small enough for one request")
            return [{'id': 0, 'content': html_content}], parallel_limit
        
        # Fast break point detection
        break_points = self.find_fast_break_points(html_content)
        ideal_chunks = min(max(2, content_length // self.target_chars), max_chunks)
        
        req_logger.log(f"Creating {ideal_chunks} large chunks for speed")
        
        chunks = []
        chars_per_chunk = content_length // ideal_chunks
        current_start = 0
        
        for chunk_id in range(ideal_chunks):
            if chunk_id == ideal_chunks - 1:
                # Last chunk gets remaining content
                chunk_content = html_content[current_start:]
                chunks.append({'id': chunk_id, 'content': chunk_content})
                req_logger.log(f"Chunk {chunk_id}: {len(chunk_content)} chars (final)")
                break
            
            ideal_end = current_start + chars_per_chunk
            best_break = self.find_nearest_break(break_points, ideal_end, current_start)
            chunk_content = html_content[current_start:best_break]
            
            if len(chunk_content.strip()) > 0:
                chunks.append({'id': chunk_id, 'content': chunk_content})
                req_logger.log(f"Chunk {chunk_id}: {len(chunk_content)} chars")
            
            current_start = best_break
        
        return chunks, parallel_limit
    
    def find_nearest_break(self, break_points, target_position, min_position):
        """Simple nearest break point finder"""
        candidates = [bp for bp in break_points if min_position < bp <= target_position + 1000]
        return min(candidates, key=lambda x: abs(x - target_position)) if candidates else target_position

# Initialize fast chunker
chunker = FastHTMLChunker()

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest, http_request: Request):
    """Fast HTML translation with aggressive optimization"""
    
    request_id = str(uuid.uuid4())[:8]
    req_logger = RequestLogger(request_id)
    
    try:
        client_ip = http_request.client.host if http_request.client else "unknown"
        req_logger.log(f"🚀 FAST MODE | {request.sourceLanguage} → {request.targetLanguage} | {len(request.content):,} chars")
        
        # Get client (now cached globally)
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available")
        
        # Fast chunking
        chunks, parallel_limit = chunker.create_fast_chunks(request.content, req_logger)
        
        # Aggressive parallel translation
        translated_chunks = await translate_html_chunks_aggressive(
            chunks, request.sourceLanguage, request.targetLanguage, 
            client, parallel_limit, req_logger
        )
        
        # Simple reassembly
        final_html = ''.join([chunk['translated_content'] for chunk in sorted(translated_chunks, key=lambda x: x['id'])])
        final_html = clean_translated_html(request.content, final_html)
        
        summary = req_logger.get_summary()
        req_logger.log(f"⚡ COMPLETE | {summary['total_processing_time']}s | {summary['chunks_successful']}/{len(chunks)} success")
        
        return {
            "translatedContent": final_html,
            "requestId": request_id,
            "processingStats": {
                "totalTime": summary['total_processing_time'],
                "chunksProcessed": len(chunks),
                "parallelWorkers": parallel_limit,
                "successful": summary['chunks_successful']
            },
            "fromCache": False
        }
        
    except Exception as e:
        req_logger.log(f"❌ FAILED: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_html_chunks_aggressive(chunks, source_lang, target_lang, client, parallel_limit, req_logger):
    """Aggressive parallel translation for maximum speed"""
    
    # High concurrency semaphore
    semaphore = asyncio.Semaphore(parallel_limit)
    req_logger.log(f"⚡ AGGRESSIVE MODE: {len(chunks)} chunks, {parallel_limit} workers")
    
    async def translate_chunk_fast(chunk):
        async with semaphore:
            start_time = time.time()
            try:
                # Simplified system prompt for speed
                system_prompt = f"Translate HTML from {source_lang} to {target_lang}. Keep ALL HTML tags unchanged. Translate only text content. Return identical structure."
                
                # Use faster model with optimized settings
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Faster than gpt-4o-mini
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0,  # Faster with 0 temperature
                    max_tokens=8000,  # Higher limit for larger chunks
                    timeout=20  # Aggressive timeout
                )
                
                translated_content = response.choices[0].message.content.strip()
                cleaned_content = clean_translated_html(chunk['content'], translated_content)
                
                processing_time = time.time() - start_time
                req_logger.log_chunk_result(chunk['id'], True, len(chunk['content']), processing_time)
                
                return {
                    'id': chunk['id'],
                    'translated_content': cleaned_content,
                    'success': True
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                req_logger.log_chunk_result(chunk['id'], False, len(chunk['content']), processing_time, str(e))
                
                # Return original content on failure
                return {
                    'id': chunk['id'],
                    'translated_content': chunk['content'],
                    'success': False
                }
    
    # Execute all translations concurrently
    tasks = [translate_chunk_fast(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions in results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            req_logger.log(f"Exception in chunk {i}: {str(result)}", "ERROR")
            final_results.append({
                'id': i,
                'translated_content': chunks[i]['content'],
                'success': False
            })
        else:
            final_results.append(result)
    
    successful = len([r for r in final_results if r['success']])
    req_logger.log(f"⚡ Parallel complete: {successful}/{len(chunks)} successful")
    
    return final_results

@langapi.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "3.0.0-fast",
        "openai_ready": bool(get_openai_client())
    }

@langapi.get("/")
async def root():
    return {
        "service": "LangAPI Fast",
        "version": "3.0.0",
        "description": "Optimized for sub-10 second HTML translation",
        "optimizations": [
            "Larger chunks (4000 chars vs 1500)",
            "Higher parallelism (up to 30 concurrent)",
            "Faster model (GPT-3.5-turbo)",
            "Simplified processing pipeline",
            "Cached OpenAI client"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
