import os 
import pdf2image
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os

# Config
PDF_PATH = "input.pdf"
OUTPUT_FOLDER = "images"
CHROMA_COLLECTION = "pdf_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_TEXT_FILE = "extracted_text.txt"
POPPLER_PATH = r"D:\Downloads\check\poppler-24.08.0\Library\bin"

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def pdf_to_images(pdf_path, output_folder):
    try:
        images = pdf2image.convert_from_path(pdf_path, poppler_path= POPPLER_PATH)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def process_image_to_text(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        text_prompt = "Act like a text scanner. Extract text as it is without analyzing it and without summarizing it."
        
        # Create properly formatted parts list
        parts = [
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_data
                }
            },
            {
                "text": text_prompt
            }
        ]
        
        # Generate content with properly formatted request
        response = model.generate_content(parts)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""

def store_in_chromadb(texts, collection_name, ids):
    try:
        chroma_client = chromadb.Client()
        gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)
        collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=gemini_ef)
        collection.upsert(documents=texts, ids=ids)
        return collection
    except Exception as e:
        print(f"Error storing data in ChromaDB: {e}")
        return None

def retrieve_and_generate(query, collection):
    try:
        results = collection.query(query_texts=[query], n_results=3)
        context = " ".join(results["documents"][0])
        response = model.generate_content(f"Answer the following question based on the context: {query}\nContext: {context}")
        return response.text.strip()
    except Exception as e:
        print(f"Error retrieving and generating response: {e}")
        return None

if __name__ == "__main__":
    image_paths = pdf_to_images(PDF_PATH, OUTPUT_FOLDER)
    if not image_paths:
        exit(1)

    extracted_text = ""
    for image_path in image_paths:
        extracted_text += process_image_to_text(image_path) + " "
    
    try:
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        print(f"Extracted text saved to {OUTPUT_TEXT_FILE}")
    except Exception as e:
        print(f"Error saving extracted text to file: {e}")

    text_chunks = [extracted_text[i:i + 1000] for i in range(0, len(extracted_text), 1000)] 
    ids = [f"chunk_{i}" for i in range(len(text_chunks))]

    collection = store_in_chromadb(text_chunks, CHROMA_COLLECTION, ids)

    if collection:
        query = "What is the main topic?"
        answer = retrieve_and_generate(query, collection)
        if answer:
            print(f"Answer: {answer}")

        #check lai
        result = collection.get(ids[0], include=['documents'])
        print(result)
