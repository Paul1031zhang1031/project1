import requests

def get_similarity_score(text1: str, text2: str, api_key: str) -> float:
    """
    Calculates the semantic similarity between two texts using the API Ninjas service.
    
    This is a generic utility function. It takes any two strings and returns a
    similarity score between 0.0 and 1.0.
    
    Returns 0.0 if an error occurs or if inputs are invalid.
    """
    # Safety check for empty or invalid input
    if not api_key or not isinstance(text1, str) or not isinstance(text2, str) or not text1.strip() or not text2.strip():
        return 0.0

    api_url = 'https://api.api-ninjas.com/v1/textsimilarity'
    headers = {'X-Api-Key': api_key}
    
    # API Ninjas has a character limit, so we truncate to be safe.
    body = {
        'text_1': text1[:4900], 
        'text_2': text2[:4900]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        result = response.json()
        return result.get('similarity', 0.0)
    except requests.exceptions.RequestException as e:
        # This handles network errors, bad status codes, etc.
        print(f"API Ninjas request error: {e}")
        return 0.0
    except Exception as e:
        # This is a catch-all for any other unexpected errors.
        print(f"An unexpected error occurred during similarity calculation: {e}")
        return 0.0