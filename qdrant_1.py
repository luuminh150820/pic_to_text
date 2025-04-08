import os
import base64
import re
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from concurrent.futures import ThreadPoolExecutor

# Config
OUTPUT_FOLDER = "images"
QDRANT_COLLECTION = "image_text_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
VECTOR_SIZE = 768
QDRANT_URL = os.environ.get("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
MAX_CHUNK_SIZE = 1000
MIN_SIMILARITY_SCORE = 0.7

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Gemini and Qdrant
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = genai.GenerativeModel("embedding-001")
qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)

def get_embedding(text):
    """Generate embedding for text"""
    try:
        result = embedding_model.generate_content(text)
        return result.embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def split_into_sentences(text):
    """Split text into sentences using regex pattern"""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    """Chunk text recursively while preserving semantic boundaries"""
    if not text:
        return []
        
    # First try to split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        elif len(paragraph) > max_chunk_size:
            # If paragraph is too large, split into sentences
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            sentences = split_into_sentences(paragraph)
            sentence_chunk = ""
            
            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) > max_chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk)
                    sentence_chunk = sentence
                else:
                    separator = " " if sentence_chunk else ""
                    sentence_chunk += separator + sentence
            
            if sentence_chunk:
                current_chunk = sentence_chunk
        else:
            separator = "\n\n" if current_chunk else ""
            current_chunk += separator + paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_image_to_text(image_path):
    """Extract text from image using Gemini vision model"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        extension = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif',
        }.get(extension, 'image/jpeg')
        
        parts = [
            {"inline_data": {"mime_type": mime_type, "data": image_data}},
            {"text": "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."}
        ]
        
        response = generation_model.generate_content(parts)
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text from {os.path.basename(image_path)}: {e}")
        return ""

def setup_qdrant_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        if QDRANT_COLLECTION not in [c.name for c in collections]:
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {QDRANT_COLLECTION}")
        return True
    except Exception as e:
        print(f"Error setting up Qdrant collection: {e}")
        return False

def store_chunks_in_qdrant(chunks, metadata):
    """Store text chunks and their embeddings in Qdrant"""
    try:
        # Get embeddings for all chunks
        points = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_image": metadata.get("source_image", "unknown"),
                            "chunk_index": i,
                            "chunk_level": metadata.get("chunk_level", "document")
                        }
                    )
                )
        
        # Upload points to Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            return len(points)
        return 0
    except Exception as e:
        print(f"Error storing data in Qdrant: {e}")
        return 0

def advanced_search(query, filters=None, min_score=MIN_SIMILARITY_SCORE, limit=5):
    """Search with filtering and score thresholds"""
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Create filter conditions
        filter_condition = None
        if filters:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )
        
        # Search with filters
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_condition
        )
        
        # Apply score threshold and return results
        return [
            {
                "text": r.payload["text"],
                "score": r.score,
                "source_image": r.payload["source_image"],
                "chunk_index": r.payload["chunk_index"]
            }
            for r in search_results if r.score >= min_score
        ]
    except Exception as e:
        print(f"Error in advanced search: {e}")
        return []

def retrieve_and_generate(query, filters=None):
    """Retrieve relevant context and generate an answer"""
    try:
        results = advanced_search(query, filters=filters)
        if not results:
            return "No relevant information found."
        
        # Combine results into context
        context = "\n\n".join([f"[From {r['source_image']}]\n{r['text']}" for r in results])
        prompt = f"Answer the following question based on the context:\nQuestion: {query}\nContext: {context}"
        
        response = generation_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error retrieving and generating response: {e}")
        return "Error generating response."

def process_images_parallel(image_paths, max_workers=4):
    """Process images in parallel for better performance"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_to_text, path): path for path in image_paths}
        for future in futures:
            path = futures[future]
            try:
                text = future.result()
                if text:
                    results[path] = text
            except Exception as e:
                print(f"Error processing {os.path.basename(path)}: {e}")
    return results

def main():
    # Find all images
    image_paths = [
        os.path.join(OUTPUT_FOLDER, filename) 
        for filename in os.listdir(OUTPUT_FOLDER) 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    if not image_paths:
        print("No images found in the 'images' folder.")
        return

    # Set up Qdrant collection
    if not setup_qdrant_collection():
        print("Failed to set up Qdrant collection.")
        return

    # Process images in parallel
    print(f"Processing {len(image_paths)} images...")
    extracted_texts = process_images_parallel(image_paths)
    
    if not extracted_texts:
        print("No text extracted from images.")
        return
    
    # Save combined text to file
    combined_text = "\n\n".join(extracted_texts.values())
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"Extracted text saved to {OUTPUT_TEXT_FILE}")
    
    # Process and store text chunks
    total_chunks = 0
    for image_path, text in extracted_texts.items():
        source_image = os.path.basename(image_path)
        print(f"Processing text from {source_image}...")
        
        # Create semantic chunks
        chunks = recursive_chunk_text(text)
        
        # Store chunks in Qdrant
        metadata = {"source_image": source_image, "chunk_level": "semantic"}
        stored_chunks = store_chunks_in_qdrant(chunks, metadata)
        total_chunks += stored_chunks
    
    print(f"Stored {total_chunks} chunks in Qdrant.")
    
    # Example query
    if total_chunks > 0:
        print("\nTesting search and retrieval:")
        query = "What is the main topic of these documents?"
        print(f"Query: {query}")
        answer = retrieve_and_generate(query)
        print(f"Answer: {answer}")
        
        # Example filtered search
        if len(image_paths) > 1:
            source_filter = {"source_image": os.path.basename(image_paths[0])}
            print(f"\nFiltered search for image {source_filter['source_image']}:")
            filtered_answer = retrieve_and_generate(query, filters=source_filter)
            print(f"Filtered answer: {filtered_answer}")

if __name__ == "__main__":
    main()