import requests
import json
import os

def google_search(query, api_key=None, cx=None, num=10):

    # Use environment variables if not provided
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cx = cx or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key:
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass as parameter.")
    if not cx:
        raise ValueError("Custom Search Engine ID is required. Set GOOGLE_SEARCH_ENGINE_ID environment variable or pass as parameter.")
    
    # API URL
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'q': query,
        'key': api_key,
        'cx': cx,
        'num': num
    }
    
    response = requests.get(url, params=params)
    
    # Check request
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def display_search_results(results):
    """
    Display search results in a readable format
    """
    if not results or 'items' not in results:
        print("No results found or error in the API response.")
        return
        
    print(f"About {results.get('searchInformation', {}).get('totalResults', 'unknown')} results")
    print("-" * 80)
    
    for i, item in enumerate(results['items'], 1):
        print(f"{i}. {item['title']}")
        print(f"   URL: {item['link']}")
        print(f"   {item.get('snippet', 'No description available')}")
        print("-" * 80)


if __name__ == "__main__":
    
    search_query = "python programming tutorials"
    
    results = google_search(search_query, api_key, cx)
    
    if results:
        display_search_results(results)
    else:
        print("Failed to retrieve search results.")
