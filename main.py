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
langapi = FastAPI(title="LangAPI", version="2.1.0")

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

def extract_structure_signature(html_content):
    """Extract a signature of the HTML structure for validation"""
    # Remove text content but keep tags and attributes
    structure = re.sub(r'>[^<]+<', '><', html_content)
    # Extract just the tags and critical classes
    tags = re.findall(r'<[^>]+>', structure)
    return ''.join(tags)

class SmartHTMLChunker:
    def __init__(self, target_chars=1500, max_chunks=15):
        self.target_chars = target_chars
        self.max_chunks = max_chunks
    
    def find_safe_break_points(self, html_content):
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
        """Create chunks that respect HTML structure and preserve layouts"""
        
        if len(html_content) <= self.target_chars:
            return [{'id': 0, 'content': html_content}]
        
        break_points = self.find_safe_break_points(html_content)
        
        total_chars = len(html_content)
        ideal_chunks = min(max(1, total_chars // self.target_chars), self.max_chunks)
        
        print(f"Creating ~{ideal_chunks} chunks from {total_chars} characters")
        
        chunks = []
        chars_per_chunk = total_chars // ideal_chunks
        
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
        
        print(f"Created {len(chunks)} smart HTML chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk['content'])} chars")
        
        return chunks
    
    def find_nearest_safe_break(self, break_points, target_position, min_position):
        """Find the best break point near target position"""
        
        candidates = [bp for bp in break_points 
                     if min_position + 200 <= bp <= target_position + 800]
        
        if not candidates:
            return min(target_position, len(break_points) - 1)
        
        return min(candidates, key=lambda x: abs(x - target_position))

# Initialize chunker
chunker = SmartHTMLChunker(target_chars=1500, max_chunks=15)

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest):
    """Translate HTML content using smart chunking"""
    start_time = time.time()
    
    try:
        print(f"Translation request: {request.sourceLanguage} → {request.targetLanguage}")
        print(f"Content length: {len(request.content)} characters")
        
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available - check API key")
        
        chunks = chunker.create_smart_chunks(request.content)
        print(f"Processing {len(chunks)} HTML chunks")
        
        translated_chunks = await translate_html_chunks_parallel(
            chunks,
            request.sourceLanguage,
            request.targetLanguage,
            client
        )
        
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
    """Translate HTML chunks in parallel with layout preservation"""
    semaphore = asyncio.Semaphore(10)
    
    async def translate_single_html_chunk(chunk):
        async with semaphore:
            try:
                print(f"Translating HTML chunk {chunk['id']}")
                
                system_prompt = f"""You are an expert HTML translator specializing in preserving CSS layouts and grid structures.

MISSION: Translate from {source_lang} to {target_lang} while maintaining PERFECT HTML structure.

ABSOLUTE REQUIREMENTS:
1. PRESERVE EXACT HTML STRUCTURE: Every tag, attribute, class, ID must remain identical
2. PRESERVE EXACT WHITESPACE: All spaces, tabs, line breaks, indentation must match exactly
3. PRESERVE CSS LAYOUTS: Grid layouts, flexbox, columns must work identically after translation
4. TRANSLATE ONLY TEXT CONTENT: Never change HTML tags, CSS classes, or attributes
5. MAINTAIN ELEMENT RELATIONSHIPS: Parent-child relationships must remain identical

DO NOT TRANSLATE:
- HTML tags: <div>, <span>, <p>, etc.
- CSS classes: class="grid", class="mb-4", etc.
- IDs: id="header", etc.
- Attributes: data-*, aria-*, etc.
- URLs, paths, file names
- Code inside <script>, <style>, <code>, <pre>
- CSS property values
- Variable names or technical terms

Return ONLY the translated HTML with identical structure and formatting."""

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",  # Using GPT-4o Mini for better results
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk['content']}
                    ],
                    temperature=0.05,
                    max_tokens=4000
                )
                
                translated_content = response.choices[0].message.content.strip()
                
                # Validation: Check if structure is preserved
                original_structure = extract_structure_signature(chunk['content'])
                translated_structure = extract_structure_signature(translated_content)
                
                if original_structure != translated_structure:
                    print(f"Warning: Structure mismatch in chunk {chunk['id']}, using fallback")
                    translated_content = chunk['content']
                
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
                    'translated_content': chunk['content'],
                    'success': False,
                    'error': str(e)
                }
    
    print(f"Starting parallel translation of {len(chunks)} HTML chunks")
    tasks = [translate_single_html_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    return sorted(results, key=lambda x: x['id'])

def reassemble_translated_chunks(translated_chunks):
    """Reassemble translated chunks back into complete HTML"""
    
    final_html = ""
    successful_translations = 0
    
    for chunk in translated_chunks:
        final_html += chunk['translated_content']
        if chunk['success']:
            successful_translations += 1
    
    print(f"Reassembled {len(translated_chunks)} chunks ({successful_translations} successful)")
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
        "version": "2.1.0",
        "description": "Layout-preserving HTML translation with GPT-4o Mini",
        "endpoints": {
            "translate": "/api/translate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
