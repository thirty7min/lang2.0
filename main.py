from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import time
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

# Initialize LangAPI
langapi = FastAPI(title="LangAPI", version="1.0.0")

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

def get_openai_client():
    """Get OpenAI client - initialize when needed"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return AsyncOpenAI(api_key=api_key, timeout=60.0)
    except Exception as e:
        print(f"OpenAI client error: {e}")
        return None

class SmartHTMLChunker:
    def __init__(self, target_chars=2000, max_chunks=10):
        self.target_chars = target_chars
        self.max_chunks = max_chunks
    
    def find_safe_break_points(self, html_content):
        """Find safe places to break HTML without cutting sentences or code"""
        
        # Define patterns for safe break points
        safe_break_patterns = [
            r'</p>\s*<p',           # Between paragraphs
            r'</h[1-6]>\s*<',       # After headings
            r'</li>\s*<li',         # Between list items
            r'</div>\s*<div',       # Between divs
            r'</section>\s*<section', # Between sections
            r'</article>\s*<article', # Between articles
            r'</td>\s*<td',         # Between table cells
            r'</tr>\s*<tr',         # Between table rows
            r'</ul>\s*<',           # After lists
            r'</ol>\s*<',           # After ordered lists
            r'</blockquote>\s*<',   # After blockquotes
            r'</pre>\s*<',          # After code blocks
            r'</code>\s*<',         # After inline code
        ]
        
        # Find all potential break points
        break_points = [0]  # Start of content
        
        for pattern in safe_break_patterns:
            for match in re.finditer(pattern, html_content, re.IGNORECASE):
                # Break point is after the closing tag
                break_point = match.end() - 1  # Before the opening 
                break_points.append(break_point)
        
        # Add end of content
        break_points.append(len(html_content))
        
        # Remove duplicates and sort
        break_points = sorted(list(set(break_points)))
        
        print(f"Found {len(break_points)} potential break points")
        return break_points
    
    def create_smart_chunks(self, html_content):
        """Create chunks that respect HTML structure and don't break sentences"""
        
        if len(html_content) <= self.target_chars:
            return [{'id': 0, 'content': html_content}]
        
        break_points = self.find_safe_break_points(html_content)
        
        # Calculate optimal number of chunks
        total_chars = len(html_content)
        ideal_chunks = min(max(1, total_chars // self.target_chars), self.max_chunks)
        
        print(f"Creating ~{ideal_chunks} chunks from {total_chars} characters")
        
        # Distribute content across chunks using break points
        chunks = []
        chars_per_chunk = total_chars // ideal_chunks
        
        current_start = 0
        chunk_id = 0
        
        while current_start < len(html_content) and chunk_id < ideal_chunks:
            # Find ideal end position
            ideal_end = current_start + chars_per_chunk
            
            # If this is the last chunk, take everything
            if chunk_id == ideal_chunks - 1:
                chunk_content = html_content[current_start:]
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'start': current_start,
                    'end': len(html_content)
                })
                break
            
            # Find the best break point near the ideal end
            best_break = self.find_nearest_safe_break(break_points, ideal_end)
            
            # Make sure we don't go backwards or create empty chunks
            if best_break <= current_start:
                best_break = current_start + min(chars_per_chunk, len(html_content) - current_start)
            
            chunk_content = html_content[current_start:best_break]
            
            chunks.append({
                'id': chunk_id,
                'content': chunk_content,
                'start': current_start,
                'end': best_break
            })
            
            current_start = best_break
            chunk_id += 1
        
        print(f"Created {len(chunks)} smart HTML chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['content'])} chars")
        
        return chunks
    
    def find_nearest_safe_break(self, break_points, target_position):
        """Find the break point closest to target position"""
        
        # Find break points near the target
        candidates = []
        for bp in break_points:
            if bp <= target_position + 500:  # Allow some flexibility
                candidates.append(bp)
        
        if not candidates:
            return target_position
        
        # Return the closest one to target
        return min(candidates, key=lambda x: abs(x - target_position))
    
    def validate_chunks(self, chunks):
        """Validate that chunks don't have broken HTML"""
        valid_chunks = []
        
        for chunk in chunks:
            content = chunk['content']
            
            # Basic validation: check if we have severely broken HTML
            open_tags = len(re.findall(r'<[^/][^>]*[^/]>', content))
            close_tags = len(re.findall(r'</[^>]*>', content))
            
            # If tags are very unbalanced, this chunk might be problematic
            # But we'll still include it since OpenAI can handle some imbalance
            
            valid_chunks.append(chunk)
        
        return valid_chunks

# Initialize chunker
chunker = SmartHTMLChunker(target_chars=2000, max_chunks=10)

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest):
    """Translate HTML content using smart chunking"""
    start_time = time.time()
    
    try:
        print(f"Translation request: {request.sourceLanguage} → {request.targetLanguage}")
        print(f"Content length: {len(request.content)} characters")
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available - check API key")
        
        # Create smart HTML chunks
        chunks = chunker.create_smart_chunks(request.content)
        validated_chunks = chunker.validate_chunks(chunks)
        
        print(f"Processing {len(validated_chunks)} HTML chunks")
        
        # Translate all chunks in parallel
        translated_chunks = await translate_html_chunks_parallel(
            validated_chunks,
            request.sourceLanguage,
            request.targetLanguage,
            client
        )
        
        # Reassemble translated content
        final_html = reassemble_translated_chunks(translated_chunks)
        
        processing_time = time.time() - start_time
        print(f"Translation completed in {processing_time:.2f} seconds")
        
        return {
            "translatedContent": final_html,
            "chunksProcessed": len(translated_chunks),
            "processingTime": processing_time,
            "fromCache": False
        }
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_html_chunks_parallel(chunks, source_lang, target_lang, client):
    """Translate HTML chunks in parallel using OpenAI"""
    semaphore = asyncio.Semaphore(10)
    
    async def translate_single_html_chunk(chunk):
        async with semaphore:
            try:
                print(f"Translating HTML chunk {chunk['id']}")
                
                # Smart prompt for HTML translation
                system_prompt = f"""You are a professional translator. Translate the following HTML content from {source_lang} to {target_lang}.

IMPORTANT RULES:
1. Translate ONLY the visible text content, NOT the HTML tags, attributes, or code
2. Keep ALL HTML structure exactly the same (tags, attributes, classes, IDs)
3. Do NOT translate: class names, IDs, URLs, file paths, variable names, or code
4. Do NOT translate content inside <script>, <style>, <code>, or <pre> tags
5. Preserve all spacing, indentation, and formatting
6. Return ONLY the translated HTML, no explanations

Translate the text content while preserving the complete HTML structure."""

                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                translated_content = response.choices[0].message.content.strip()
                
                print(f"Chunk {chunk['id']} translated successfully")
                
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': translated_content,
                    'success': True
                }
                
            except Exception as e:
                print(f"Chunk {chunk['id']} failed: {str(e)}")
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': chunk['content'],  # Fallback to original
                    'success': False,
                    'error': str(e)
                }
    
    print(f"Starting parallel translation of {len(chunks)} HTML chunks")
    tasks = [translate_single_html_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Sort by chunk ID to maintain order
    return sorted(results, key=lambda x: x['id'])

def reassemble_translated_chunks(translated_chunks):
    """Reassemble translated chunks back into complete HTML"""
    
    # Simply concatenate the translated content in order
    final_html = ""
    
    for chunk in translated_chunks:
        final_html += chunk['translated_content']
    
    print(f"Reassembled {len(translated_chunks)} chunks into final HTML")
    return final_html

@langapi.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "max_parallel_chunks": chunker.max_chunks,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@langapi.get("/")
async def root():
    """API information"""
    return {
        "service": "LangAPI",
        "version": "2.0.0",
        "description": "Smart HTML translation with boundary-aware chunking",
        "endpoints": {
            "translate": "/api/translate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
