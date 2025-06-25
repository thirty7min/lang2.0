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
langapi = FastAPI(title="LangAPI", version="2.3.0")

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
        self.log(f"Chunk {chunk_id}: {status} | {chars} chars | {result['processing_time_ms']}ms | {error if error else 'OK'}")
    
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

def get_openai_client():
    """Get OpenAI client - initialize when needed"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(api_key=api_key, timeout=60.0)
    except Exception as e:
        logger.error(f"OpenAI client error: {e}")
        return None

def clean_translated_html(original_content, translated_content):
    """Clean up translated HTML to remove extra wrapper tags"""
    
    unwanted_wrappers = [
        r'^<html[^>]*>(.*)</html>$',
        r'^<body[^>]*>(.*)</body>$', 
        r'^<div[^>]*>(.*)</div>$',
        r'^<p[^>]*>(.*)</p>$'
    ]
    
    cleaned = translated_content.strip()
    
    for pattern in unwanted_wrappers:
        match = re.match(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match and not re.match(pattern, original_content.strip(), re.DOTALL | re.IGNORECASE):
            cleaned = match.group(1).strip()
    
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned

class SmartHTMLChunker:
    def __init__(self, target_chars=1500):
        self.target_chars = target_chars
    
    def calculate_optimal_config(self, content_length, req_logger):
        """Calculate optimal chunks and parallel processing based on content size"""
        
        if content_length < 3000:
            config = (2, 2, "Small content")
        elif content_length < 8000:
            config = (5, 4, "Medium content")
        elif content_length < 20000:
            config = (10, 6, "Large content")
        elif content_length < 50000:
            config = (15, 8, "Very large content")
        else:
            config = (20, 10, "Huge content")
        
        max_chunks, parallel_limit, size_category = config
        req_logger.log(f"Content analysis: {content_length} chars classified as '{size_category}'")
        req_logger.log(f"Optimization: Max {max_chunks} chunks, {parallel_limit} parallel workers")
        
        return max_chunks, parallel_limit
    
    def find_safe_break_points(self, html_content, req_logger):
        """Find safe places to break HTML without cutting sentences or breaking layouts"""
        
        safe_break_patterns = [
            r'</p>\s*<p',
            r'</h[1-6]>\s*<',
            r'</li>\s*<li',
            r'</ul>\s*<',
            r'</ol>\s*<',
            r'</div>\s*</div>\s*<div',
            r'</section>\s*<section',
            r'</article>\s*<article',
            r'</td>\s*<td',
            r'</tr>\s*<tr',
            r'</thead>\s*<tbody',
            r'</tbody>\s*</table',
            r'</table>\s*<',
            r'</blockquote>\s*<',
            r'</pre>\s*<',
            r'</code>\s*<',
            r'</ul>\s*</div>\s*<div',
            r'</div>\s*<div\s+class="[^"]*col',
            r'</div>\s*<div\s+class="[^"]*grid',
        ]
        
        break_points = [0]
        pattern_matches = {}
        
        for pattern in safe_break_patterns:
            matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
            pattern_matches[pattern] = len(matches)
            for match in matches:
                break_point = match.end() - len(match.group().split('<')[-1]) - 1
                if break_point > 0:
                    break_points.append(break_point)
        
        break_points.append(len(html_content))
        break_points = sorted(list(set(break_points)))
        
        req_logger.log(f"Break point analysis: Found {len(break_points)} potential break points")
        for pattern, count in pattern_matches.items():
            if count > 0:
                req_logger.log(f"  Pattern '{pattern[:20]}...': {count} matches")
        
        return break_points
    
    def create_smart_chunks(self, html_content, req_logger):
        """Create adaptive chunks based on content size"""
        
        content_length = len(html_content)
        max_chunks, parallel_limit = self.calculate_optimal_config(content_length, req_logger)
        
        if content_length <= self.target_chars:
            req_logger.log("Single chunk strategy: Content fits in one chunk")
            return [{'id': 0, 'content': html_content}], parallel_limit
        
        break_points = self.find_safe_break_points(html_content, req_logger)
        ideal_chunks = min(max(1, content_length // self.target_chars), max_chunks)
        
        req_logger.log(f"Chunking strategy: Creating {ideal_chunks} chunks from {content_length} chars")
        
        chunks = []
        chars_per_chunk = content_length // ideal_chunks
        current_start = 0
        chunk_id = 0
        
        while current_start < len(html_content) and chunk_id < ideal_chunks:
            ideal_end = current_start + chars_per_chunk
            
            if chunk_id == ideal_chunks - 1:
                chunk_content = html_content[current_start:]
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'start': current_start,
                    'end': len(html_content)
                })
                req_logger.log(f"Chunk {chunk_id}: {len(chunk_content)} chars (final chunk)")
                break
            
            best_break = self.find_nearest_safe_break(break_points, ideal_end, current_start)
            chunk_content = html_content[current_start:best_break]
            
            if len(chunk_content.strip()) > 0:
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'start': current_start,
                    'end': best_break
                })
                req_logger.log(f"Chunk {chunk_id}: {len(chunk_content)} chars (pos {current_start}-{best_break})")
                chunk_id += 1
            
            current_start = best_break
        
        req_logger.log(f"Chunking complete: Created {len(chunks)} chunks, will use {parallel_limit} parallel workers")
        return chunks, parallel_limit
    
    def find_nearest_safe_break(self, break_points, target_position, min_position):
        """Find the best break point near target position"""
        
        candidates = [bp for bp in break_points 
                     if min_position + 200 <= bp <= target_position + 800]
        
        if not candidates:
            return min(target_position, len(break_points) - 1)
        
        return min(candidates, key=lambda x: abs(x - target_position))

# Initialize chunker
chunker = SmartHTMLChunker(target_chars=1500)

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest, http_request: Request):
    """Translate HTML content using adaptive smart chunking with detailed logging"""
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    req_logger = RequestLogger(request_id)
    
    try:
        # Log request details
        client_ip = http_request.client.host if http_request.client else "unknown"
        req_logger.log(f"🚀 NEW REQUEST | {request.sourceLanguage} → {request.targetLanguage} | IP: {client_ip}")
        req_logger.log(f"Content length: {len(request.content):,} characters")
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            req_logger.log("❌ FAILED: OpenAI client not available", "ERROR")
            raise HTTPException(status_code=500, detail="OpenAI client not available - check API key")
        
        req_logger.log("✅ OpenAI client initialized")
        
        # Create adaptive chunks
        chunking_start = time.time()
        chunks, parallel_limit = chunker.create_smart_chunks(request.content, req_logger)
        chunking_time = time.time() - chunking_start
        req_logger.log(f"⚡ Chunking completed in {chunking_time*1000:.1f}ms")
        
        # Translate with adaptive parallel processing
        translation_start = time.time()
        translated_chunks = await translate_html_chunks_parallel(
            chunks,
            request.sourceLanguage,
            request.targetLanguage,
            client,
            parallel_limit,
            req_logger
        )
        translation_time = time.time() - translation_start
        req_logger.log(f"🔄 Translation phase completed in {translation_time:.2f}s")
        
        # Reassemble and clean translated content
        assembly_start = time.time()
        final_html = reassemble_translated_chunks(translated_chunks, request.content, req_logger)
        assembly_time = time.time() - assembly_start
        req_logger.log(f"🔧 Assembly completed in {assembly_time*1000:.1f}ms")
        
        # Generate summary
        summary = req_logger.get_summary()
        req_logger.log(f"✅ REQUEST COMPLETE | Total: {summary['total_processing_time']}s | Success: {summary['chunks_successful']}/{len(chunks)}")
        
        return {
            "translatedContent": final_html,
            "requestId": request_id,
            "processingStats": {
                "chunksProcessed": len(translated_chunks),
                "chunksSuccessful": summary['chunks_successful'],
                "chunksFailed": summary['chunks_failed'],
                "parallelWorkers": parallel_limit,
                "totalProcessingTime": summary['total_processing_time'],
                "phases": {
                    "chunking": round(chunking_time * 1000, 1),
                    "translation": round(translation_time * 1000, 1),
                    "assembly": round(assembly_time * 1000, 1)
                }
            },
            "chunkDetails": summary['chunk_details'],
            "fromCache": False
        }
        
    except Exception as e:
        req_logger.log(f"❌ REQUEST FAILED: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_html_chunks_parallel(chunks, source_lang, target_lang, client, parallel_limit, req_logger):
    """Translate HTML chunks with adaptive parallel processing and detailed logging"""
    
    semaphore = asyncio.Semaphore(parallel_limit)
    req_logger.log(f"🔄 Starting parallel translation: {len(chunks)} chunks, {parallel_limit} workers")
    
    async def translate_single_html_chunk(chunk):
        chunk_start = time.time()
        async with semaphore:
            try:
                req_logger.log(f"🔄 Processing chunk {chunk['id']} ({len(chunk['content'])} chars)")
                
                system_prompt = f"""You are an expert HTML translator. Translate from {source_lang} to {target_lang}.

CRITICAL RULES:
1. Return EXACTLY the same HTML structure as input - no additions, no wrapper tags
2. Translate ONLY the visible text content between HTML tags
3. Keep ALL HTML tags, attributes, classes, and IDs exactly as they are
4. Do NOT add <html>, <body>, <div> or any wrapper tags that weren't in the input
5. Preserve exact spacing, line breaks, and indentation
6. Do NOT translate technical terms, CSS classes, or code

If input starts with <div>, output should start with <div>
If input starts with <p>, output should start with <p>
If input starts with text, output should start with text

NEVER add wrapper tags around the content.

Return only the translated version with identical structure."""

                api_call_start = time.time()
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0.02,
                    max_tokens=4000
                )
                api_call_time = time.time() - api_call_start
                
                translated_content = response.choices[0].message.content.strip()
                cleaned_content = clean_translated_html(chunk['content'], translated_content)
                
                chunk_time = time.time() - chunk_start
                req_logger.log_chunk_result(
                    chunk['id'], 
                    True, 
                    len(chunk['content']), 
                    chunk_time
                )
                req_logger.log(f"  API call took {api_call_time*1000:.0f}ms")
                
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': cleaned_content,
                    'success': True,
                    'processing_time': chunk_time,
                    'api_call_time': api_call_time
                }
                
            except Exception as e:
                chunk_time = time.time() - chunk_start
                error_msg = str(e)
                req_logger.log_chunk_result(
                    chunk['id'], 
                    False, 
                    len(chunk['content']), 
                    chunk_time, 
                    error_msg
                )
                
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': chunk['content'],
                    'success': False,
                    'error': error_msg,
                    'processing_time': chunk_time
                }
    
    # Execute all translations
    tasks = [translate_single_html_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Log summary of parallel execution
    successful = len([r for r in results if r['success']])
    failed = len([r for r in results if not r['success']])
    avg_time = sum([r['processing_time'] for r in results]) / len(results)
    
    req_logger.log(f"🏁 Parallel execution complete: {successful} success, {failed} failed, avg {avg_time*1000:.0f}ms per chunk")
    
    return sorted(results, key=lambda x: x['id'])

def reassemble_translated_chunks(translated_chunks, original_content, req_logger):
    """Reassemble translated chunks and ensure no extra HTML"""
    
    req_logger.log(f"🔧 Reassembling {len(translated_chunks)} chunks")
    
    final_html = ""
    successful_translations = 0
    
    for chunk in translated_chunks:
        final_html += chunk['translated_content']
        if chunk['success']:
            successful_translations += 1
    
    # Final cleanup
    final_html = clean_translated_html(original_content, final_html)
    
    original_size = len(original_content)
    final_size = len(final_html)
    size_change = ((final_size - original_size) / original_size) * 100
    
    req_logger.log(f"📊 Assembly stats: {successful_translations}/{len(translated_chunks)} successful")
    req_logger.log(f"📏 Size change: {original_size:,} → {final_size:,} chars ({size_change:+.1f}%)")
    
    return final_html

@langapi.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@langapi.get("/")
async def root():
    """API information"""
    return {
        "service": "LangAPI",
        "version": "2.3.0",
        "description": "Adaptive HTML translation with comprehensive logging",
        "features": [
            "Detailed request tracking with unique IDs",
            "Chunk-by-chunk success/failure logging",
            "Performance timing for all phases",
            "Adaptive scaling and optimization logs"
        ],
        "endpoints": {
            "translate": "/api/translate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
