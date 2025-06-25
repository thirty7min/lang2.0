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

# Minimal logging for performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("LangAPI")

langapi = FastAPI(title="LangAPI", version="2.4.0")

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

class FastRequestLogger:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.chunk_results = []
    
    def log_chunk_result(self, chunk_id: int, success: bool, chars: int, processing_time: float):
        # Minimal logging for speed
        self.chunk_results.append({
            "chunk_id": chunk_id,
            "success": success,
            "characters": chars,
            "processing_time_ms": round(processing_time * 1000, 1)
        })
    
    def get_summary(self):
        total_time = time.time() - self.start_time
        successful = len([r for r in self.chunk_results if r['success']])
        failed = len(self.chunk_results) - successful
        
        return {
            "request_id": self.request_id,
            "total_time": round(total_time, 2),
            "chunks_successful": successful,
            "chunks_failed": failed,
            "chunk_details": self.chunk_results
        }

def get_openai_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(api_key=api_key, timeout=30.0)  # Shorter timeout
    except Exception as e:
        logger.error(f"OpenAI client error: {e}")
        return None

class AggressiveChunker:
    def __init__(self, target_chars=800):  # Much smaller chunks
        self.target_chars = target_chars
    
    def calculate_optimal_config(self, content_length):
        """Aggressive parallel processing for speed"""
        
        if content_length < 2000:      # Small
            return 3, 3
        elif content_length < 5000:    # Medium  
            return 8, 8
        elif content_length < 15000:   # Large (like iPhone article)
            return 20, 15  # Much more aggressive!
        elif content_length < 30000:   # Very large
            return 30, 20
        else:                          # Huge
            return 40, 25
    
    def find_safe_break_points(self, html_content):
        """Fast break point detection"""
        
        # Simplified patterns for speed
        patterns = [
            r'</p>\s*<p', r'</h[1-6]>\s*<', r'</li>\s*<li',
            r'</ul>\s*<', r'</div>\s*</div>\s*<div', r'</section>\s*<section'
        ]
        
        break_points = [0]
        for pattern in patterns:
            for match in re.finditer(pattern, html_content, re.IGNORECASE):
                break_points.append(match.end() - 1)
        
        break_points.append(len(html_content))
        return sorted(list(set(break_points)))
    
    def create_smart_chunks(self, html_content, req_logger):
        """Fast chunking optimized for parallel processing"""
        
        content_length = len(html_content)
        max_chunks, parallel_limit = self.calculate_optimal_config(content_length)
        
        logger.info(f"[{req_logger.request_id}] Fast chunking: {content_length} chars → {max_chunks} chunks, {parallel_limit} parallel")
        
        if content_length <= self.target_chars:
            return [{'id': 0, 'content': html_content}], parallel_limit
        
        # Fast chunking algorithm
        break_points = self.find_safe_break_points(html_content)
        
        # Create more, smaller chunks for faster processing
        actual_chunks = min(max_chunks, content_length // self.target_chars + 1)
        chars_per_chunk = content_length // actual_chunks
        
        chunks = []
        current_start = 0
        
        for chunk_id in range(actual_chunks):
            if chunk_id == actual_chunks - 1:
                # Last chunk gets everything remaining
                chunk_content = html_content[current_start:]
            else:
                ideal_end = current_start + chars_per_chunk
                # Find nearest safe break
                best_break = min(break_points, key=lambda x: abs(x - ideal_end) if x > current_start + 100 else float('inf'))
                chunk_content = html_content[current_start:best_break]
            
            if len(chunk_content.strip()) > 0:
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content
                })
            
            current_start = best_break if chunk_id < actual_chunks - 1 else len(html_content)
        
        logger.info(f"[{req_logger.request_id}] Created {len(chunks)} chunks (avg {content_length//len(chunks)} chars each)")
        return chunks, parallel_limit

# Initialize aggressive chunker
chunker = AggressiveChunker(target_chars=800)

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest, http_request: Request):
    """Ultra-fast translation targeting sub-10 seconds"""
    
    request_id = str(uuid.uuid4())[:8]
    req_logger = FastRequestLogger(request_id)
    
    try:
        logger.info(f"[{request_id}] FAST REQUEST: {request.sourceLanguage}→{request.targetLanguage} | {len(request.content)} chars")
        
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available")
        
        # Ultra-fast chunking
        chunks, parallel_limit = chunker.create_smart_chunks(request.content, req_logger)
        
        # Aggressive parallel translation
        translated_chunks = await translate_ultra_fast(
            chunks, request.sourceLanguage, request.targetLanguage, client, parallel_limit, req_logger
        )
        
        # Fast reassembly
        final_html = fast_reassemble(translated_chunks)
        
        summary = req_logger.get_summary()
        logger.info(f"[{request_id}] COMPLETE: {summary['total_time']}s | {summary['chunks_successful']}/{len(chunks)} success")
        
        return {
            "translatedContent": final_html,
            "requestId": request_id,
            "stats": {
                "totalTime": summary['total_time'],
                "chunksProcessed": len(chunks),
                "parallelWorkers": parallel_limit,
                "avgChunkTime": round(sum(r['processing_time_ms'] for r in summary['chunk_details']) / len(summary['chunk_details']), 1) if summary['chunk_details'] else 0
            },
            "chunkDetails": summary['chunk_details']
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_ultra_fast(chunks, source_lang, target_lang, client, parallel_limit, req_logger):
    """Ultra-aggressive parallel translation"""
    
    # Much higher concurrency
    semaphore = asyncio.Semaphore(parallel_limit)
    logger.info(f"[{req_logger.request_id}] PARALLEL START: {len(chunks)} chunks, {parallel_limit} workers")
    
    async def translate_chunk_fast(chunk):
        chunk_start = time.time()
        async with semaphore:
            try:
                # Minimal, fast prompt
                system_prompt = f"""Translate HTML from {source_lang} to {target_lang}. Keep exact HTML structure. Translate only text content, not tags/attributes."""
                
                response = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0.0,  # Fastest setting
                    max_tokens=2000   # Smaller limit for speed
                )
                
                translated_content = response.choices[0].message.content.strip()
                chunk_time = time.time() - chunk_start
                
                req_logger.log_chunk_result(chunk['id'], True, len(chunk['content']), chunk_time)
                
                return {
                    'id': chunk['id'],
                    'content': translated_content,
                    'success': True
                }
                
            except Exception as e:
                chunk_time = time.time() - chunk_start
                req_logger.log_chunk_result(chunk['id'], False, len(chunk['content']), chunk_time)
                
                return {
                    'id': chunk['id'],
                    'content': chunk['content'],  # Fallback to original
                    'success': False
                }
    
    # Execute all chunks simultaneously
    tasks = [translate_chunk_fast(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    clean_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            clean_results.append({
                'id': i,
                'content': chunks[i]['content'],
                'success': False
            })
        else:
            clean_results.append(result)
    
    return sorted(clean_results, key=lambda x: x['id'])

def fast_reassemble(translated_chunks):
    """Fast reassembly without heavy processing"""
    final_html = ""
    for chunk in translated_chunks:
        final_html += chunk['content']
    return final_html

@langapi.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.4.0 - Ultra Fast"}

@langapi.get("/")
async def root():
    return {
        "service": "LangAPI Ultra Fast",
        "version": "2.4.0",
        "target": "Sub-10 second translations",
        "optimizations": [
            "Small chunks (800 chars) for faster GPT processing",
            "Aggressive parallelism (15-25 workers for large content)",
            "Minimal logging overhead",
            "Simplified prompts for speed",
            "Fast reassembly without heavy cleaning"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
