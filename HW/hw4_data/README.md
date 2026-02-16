# HW4 Data Directory

This directory is for storing HTML files that will be used in the HW 4 RAG (Retrieval-Augmented Generation) chatbot.

## Instructions

1. **Add HTML files**: Place all your HTML files in this directory
2. **File format**: Files should have the `.html` extension
3. **Processing**: The application will automatically:
   - Extract text content from HTML files
   - Remove script and style elements
   - Split content into chunks (1000 characters with 200 character overlap)
   - Create embeddings using OpenAI's text-embedding-3-small model
   - Store in ChromaDB vector database

## How to Add Files

Simply copy your HTML files to this directory. For example:
```bash
cp /path/to/your/html/files/*.html /workspaces/IST-488-HW/HW/hw4_data/
```

The application will automatically process them when you run HW 4.

## Supported Content

The HTML parser will extract clean text from:
- Headings
- Paragraphs
- Lists
- Tables
- And other HTML elements

Scripts and styles are automatically removed during processing.
