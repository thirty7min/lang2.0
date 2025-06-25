from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import time
import json
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
import re

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
        return AsyncOpenAI(api_key=api_key)
    except Exception as e:
        print(f"OpenAI client error: {e}")
        return None

class SmartChunker:
    def __init__(self, target_chars=1000, max_chunks=10):
        self.target_chars = target_chars
        self.max_chunks = max_chunks
    
    def extract_translatable_text(self, html_content):
        """Extract only text that should be translated"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove elements that shouldn't be translated
        for element in soup(['script', 'style', 'noscript', 'meta', 'link', 'code']):
            element.decompose()
        
        translatable_elements = []
        text_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div', 'li', 'td', 'th', 'a', 'button']
        
        for element in soup.find_all(text_tags):
            text = element.get_text(strip=True)
            if text and len(text) > 3 and not self.looks_like_code(text):
                translatable_elements.append({
                    'text': text,
                    'tag': element.name,
                    'selector': self.generate_selector(element)
                })
        
        return translatable_elements, soup
    
    def looks_like_code(self, text):
        """Check if text looks like code and shouldn't be translated"""
        code_patterns = [
            r'^\s*[{}\[\]();]+\s*$',
            r'^\s*function\s+\w+',
            r'^\s*class\s+\w+',
            r'^\s*import\s+',
            r'^\s*<[^>]+>\s*$',
            r'^\s*\/\*.*\*\/\s*$',
        ]
        
        for pattern in code_patterns:
            if re.match(pattern, text):
                return True
        
        special_chars = len(re.findall(r'[{}()\[\];=<>]', text))
        return len(text) > 0 and special_chars / len(text) > 0.3
    
    def generate_selector(self, element):
        """Generate a simple selector for the element"""
        if element.get('id'):
            return f"#{element['id']}"
        
        tag = element.name
        if element.get('class'):
            classes = element['class']
            return f"{tag}.{classes[0]}" if classes else tag
        
        return tag
    
    def create_smart_chunks(self, elements):
        """Split elements into optimal chunks for parallel processing"""
        if not elements:
            return []
        
        total_chars = sum(len(elem['text']) for elem in elements)
        optimal_chunks = min(max(1, total_chars // self.target_chars), self.max_chunks)
        
        print(f"Creating {optimal_chunks} chunks from {total_chars} characters")
        
        chunk_size = len(elements) // optimal_chunks if optimal_chunks > 0 else len(elements)
        chunks = []
        
        for i in range(0, len(elements), max(1, chunk_size)):
            chunk_elements = elements[i:i + chunk_size] if chunk_size > 0 else [elements[i]]
            chunk_text = '\n'.join([elem['text'] for elem in chunk_elements])
            
            chunks.append({
                'id': len(chunks),
                'elements': chunk_elements,
                'text': chunk_text
            })
        
        return chunks[:optimal_chunks]

# Initialize chunker
chunker = SmartChunker(target_chars=1000, max_chunks=10)

@langapi.post("/api/translate")
async def translate_content(request: TranslationRequest):
    """Translate content using parallel processing"""
    start_time = time.time()
    
    try:
        print(f"Translation request: {request.sourceLanguage} → {request.targetLanguage}")
        print(f"Content length: {len(request.content)} characters")
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not available - check API key")
        
        # Extract translatable elements
        elements, soup = chunker.extract_translatable_text(request.content)
        
        if not elements:
            return {
                "translatedContent": request.content,
                "message": "No translatable content found",
                "processingTime": time.time() - start_time
            }
        
        print(f"Found {len(elements)} translatable elements")
        
        # Create smart chunks
        chunks = chunker.create_smart_chunks(elements)
        print(f"Created {len(chunks)} chunks for parallel processing")
        
        # Translate all chunks in parallel
        translated_chunks = await translate_chunks_parallel(
            chunks, 
            request.sourceLanguage, 
            request.targetLanguage,
            client
        )
        
        # Apply translations back to original HTML
        final_html = apply_translations_to_html(request.content, translated_chunks)
        
        processing_time = time.time() - start_time
        print(f"Translation completed in {processing_time:.2f} seconds")
        
        return {
            "translatedContent": final_html,
            "chunksProcessed": len(chunks),
            "elementsTranslated": len(elements),
            "processingTime": processing_time,
            "fromCache": False
        }
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def translate_chunks_parallel(chunks, source_lang, target_lang, client):
    """Translate multiple chunks simultaneously"""
    semaphore = asyncio.Semaphore(10)
    
    async def translate_single_chunk(chunk):
        async with semaphore:
            try:
                print(f"Translating chunk {chunk['id']}")
                
                # NEW OpenAI v1.0+ syntax
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Translate the following text from {source_lang} to {target_lang}. "
                                     f"Return ONLY the translated text, maintaining the same structure and meaning."
                        },
                        {
                            "role": "user",
                            "content": chunk['text']
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                translated_text = response.choices[0].message.content.strip()
                translated_lines = translated_text.split('\n')
                
                translated_elements = []
                for i, element in enumerate(chunk['elements']):
                    if i < len(translated_lines):
                        translated_elements.append({
                            **element,
                            'translatedText': translated_lines[i].strip()
                        })
                    else:
                        translated_elements.append({
                            **element,
                            'translatedText': element['text']
                        })
                
                print(f"Chunk {chunk['id']} completed")
                return translated_elements
                
            except Exception as e:
                print(f"Chunk {chunk['id']} failed: {str(e)}")
                return [{**elem, 'translatedText': elem['text']} for elem in chunk['elements']]
    
    print(f"Starting parallel translation of {len(chunks)} chunks")
    tasks = [translate_single_chunk(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)
    
    all_translated_elements = []
    for chunk_result in chunk_results:
        all_translated_elements.extend(chunk_result)
    
    return all_translated_elements

def apply_translations_to_html(original_html, translated_elements):
    """Apply translations back to the original HTML structure"""
    soup = BeautifulSoup(original_html, 'html.parser')
    
    translation_map = {}
    for element in translated_elements:
        if 'translatedText' in element:
            translation_map[element['text']] = element['translatedText']
    
    updated_count = 0
    for original_text, translated_text in translation_map.items():
        for text_node in soup.find_all(text=True):
            if text_node.strip() == original_text.strip():
                text_node.replace_with(translated_text)
                updated_count += 1
    
    print(f"Applied {updated_count} translations to HTML")
    return str(soup)

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
        "version": "1.0.0",
        "endpoints": {
            "translate": "/api/translate",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(langapi, host="0.0.0.0", port=8000)
