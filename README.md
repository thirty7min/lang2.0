# LangAPI

High-performance translation service with parallel processing capabilities.

## Features

- Smart content chunking with word boundary protection
- Parallel translation processing (10 simultaneous chunks)
- HTML structure preservation
- Code block detection and protection
- Automatic scaling for content of any size

## API Endpoints

### Translate Content
```http
POST /api/translate
Content-Type: application/json

{
  "content": "<html>Your content here</html>",
  "sourceLanguage": "en",
  "targetLanguage": "es"
}
