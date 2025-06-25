from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import time
import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

# Initialize LangAPI
langapi = FastAPI(title="LangAPI", version="2.2.0")

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

def clean_translated_html(original_content, translated_content):
    """Clean up translated HTML to remove extra wrapper tags"""
    
    # Remove common wrapper tags that OpenAI might add
    unwanted_wrappers = [
        r'^<html[^>]*>(.*)</html>$',
        r'^<body[^>]*>(.*)</body>$', 
        r'^<div[^>]*>(.*)</div>$',
        r'^<p[^>]*>(.*)</p>$'
    ]
    
    cleaned = translated_content.strip()
    
    # Remove unwanted wrappers that weren't in original
    for pattern in unwanted_wrappers:
        match = re.match(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match and not re.match(pattern, original_content.strip(), re.DOTALL | re.IGNORECASE):
            cleaned = match.group(1).strip()
    
    # Remove extra whitespace but preserve structure
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    
    return cleaned

class SmartHTMLChunker:
    def __init__(self, target_chars=1500):
        self.target_chars = target_chars
    
    def calculate_optimal_config(self, content_length):
        """Calculate optimal chunks and parallel processing based on content size"""
        
        if content_length < 3000:      # Small: ~2 pages
            return 2, 2   # max_chunks, parallel_limit
        elif content_length < 8000:    # Medium: ~5 pages
            return 5, 4
        elif content_length < 20000:   # Large: ~10 pages  
            return 10, 6
        elif content_length < 50000:   # Very large: ~25 pages
            return 15, 8
        else:                          # Huge: 25+ pages
            return 20, 10
    
    def find_safe_break_points(self, html_content):
        """Find safe places to break HTML without cutting sentences or breaking layouts"""
        
        safe_break_patterns = [
            r'</p>\s*<p',                    # Between paragraphs
            r'</h[1-6]>\s*<',                # After headings
            r'</li>\s*<li',                  # Between list items
            r'</ul>\s*<',                    # After unordered lists
            r'</ol>\s*<',                    # After ordered lists
            r'</div>\s*</div>\s*<div',       # Between major div sections
            r'</section>\s*<section',        # Between sections
            r'</article>\s*<article',        # Between articles
            r'</td>\s*<td',                  # Between table cells
            r'</tr>\s*<tr',                  # Between table rows
            r'</thead>\s*<tbody',            # Between table sections
            r'</tbody>\s*</table',           # End of tables
            r'</table>\s*<',                 # After tables
            r'</blockquote>\s*<',            # After blockquotes
            r'</pre>\s*<',                   # After code blocks
            r'</code>\s*<',                  # After inline code
            r'</ul>\s*</div>\s*<div',        # Between grid columns
            r'</div>\s*<div\s+class="[^"]*col', # Before new columns
            r'</div>\s*<div\s+class="[^"]*grid', # Before new grids
        ]
        
        break_points = [0]
        
        for pattern in safe_break_patterns:
            for match in re.finditer(pattern, html_content, re.IGNORECASE):
                break_point = match.end() - len(match.group().split('<')[-1]) - 1
                if break_point > 0:
                    break_points.append(break_point)
        
        break_points.append(len(html_content))
        break_points = sorted(list(set(break_points)))
        
        print(f"Found {len(break_points)} potential break points")
        return break_points
    
    def create_smart_chunks(self, html_content):
        """Create adaptive chunks based on content size"""
        
        content_length = len(html_content)
        max_chunks, parallel_limit = self.calculate_optimal_config(content_length)
        
        print(f"Content: {content_length} chars → Max {max_chunks} chunks, {parallel_limit} parallel")
        
        if content_length <= self.target_chars:
            return [{'id': 0, 'content': html_content}], parallel_limit
        
        break_points = self.find_safe_break_points(html_content)
        
        # Calculate ideal chunks based on content and limits
        ideal_chunks = min(max(1, content_length // self.target_chars), max_chunks)
        
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
                chunk_id += 1
            
            current_start = best_break
        
        print(f"Created {len(chunks)} adaptive chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['content'])} chars")
        
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
async def translate_content(request: TranslationRequest):
    """Translate HTML content using adaptive smart chunking"""
    start_time = time.time()
    
    try:
        print(f"Translation request: {request.sourceLanguage} → {request.targetLanguage}")
        print(f"Content length: {len(request.content)} characters")
        
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available - check API key")
        
        # Create adaptive chunks
        chunks, parallel_limit = chunker.create_smart_chunks(request.content)
        print(f"Processing {len(chunks)} chunks with {parallel_limit} parallel workers")
        
        # Translate with adaptive parallel processing
        translated_chunks = await translate_html_chunks_parallel(
            chunks,
            request.sourceLanguage,
            request.targetLanguage,
            client,
            parallel_limit
        )
        
        # Reassemble and clean translated content
        final_html = reassemble_translated_chunks(translated_chunks, request.content)
        
        processing_time = time.time() - start_time
        print(f"Translation completed in {processing_time:.2f} seconds")
        
        return {
            "translatedContent": final_html,
            "chunksProcessed": len(translated_chunks),
            "parallelWorkers": parallel_limit,
            "processingTime": processing_time,
            "fromCache": False
        }
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_html_chunks_parallel(chunks, source_lang, target_lang, client, parallel_limit):
    """Translate HTML chunks with adaptive parallel processing"""
    semaphore = asyncio.Semaphore(parallel_limit)
    
    async def translate_single_html_chunk(chunk):
        async with semaphore:
            try:
                print(f"Translating chunk {chunk['id']} ({len(chunk['content'])} chars)")
                
                # Enhanced prompt to prevent HTML wrapper addition
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

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0.02,  # Very low temperature for consistency
                    max_tokens=4000
                )
                
                translated_content = response.choices[0].message.content.strip()
                
                # Clean up any unwanted wrapper tags
                cleaned_content = clean_translated_html(chunk['content'], translated_content)
                
                print(f"Chunk {chunk['id']} completed")
                
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': cleaned_content,
                    'success': True
                }
                
            except Exception as e:
                print(f"Chunk {chunk['id']} failed: {str(e)}")
                return {
                    'id': chunk['id'],
                    'original_content': chunk['content'],
                    'translated_content': chunk['content'],
                    'success': False,
                    'error': str(e)
                }
    
    print(f"Starting parallel translation with {parallel_limit} workers")
    tasks = [translate_single_html_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    return sorted(results, key=lambda x: x['id'])

def reassemble_translated_chunks(translated_chunks, original_content):
    """Reassemble translated chunks and ensure no extra HTML"""
    
    final_html = ""
    successful_translations = 0
    
    for chunk in translated_chunks:
        final_html += chunk['translated_content']
        if chunk['success']:
            successful_translations += 1
    
    # Final cleanup to remove any remaining wrapper issues
    final_html = clean_translated_html(original_content, final_html)
    
    print(f"Reassembled {len(translated_chunks)} chunks ({successful_translations} successful)")
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
        "version": "2.2.0",
        "description": "Adaptive HTML translation with clean output",
        "features": [
            "Adaptive chunking based on content size",
            "Smart parallel processing scaling", 
            "Clean HTML output without wrapper tags",
            "Layout-preserving translation"
        ],
        "endpoints": {
            "translate": "/api/translate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
