import spacy
import json
from collections import Counter
from string import punctuation

def extract_keywords_to_json(text, top_n=10):
    """
    Extract keywords from text and return them as JSON.
    
    Parameters:
        text (str): The input text to extract keywords from
        top_n (int): The number of top keywords to return
        
    Returns:
        str: A JSON string containing the keywords and their counts
    """
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        # If model is not found, handle the error
        return json.dumps({"error": "Required spaCy model not found. Install with: python -m spacy download en_core_web_sm"})
    
    # Extract keywords
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    
    # Count keywords
    word_counts = Counter(result)
    most_common_list = word_counts.most_common(top_n)
    
    # Create result dictionary
    keywords_list = [{"word": word, "count": count} for word, count in most_common_list]
    result_dict = {"keywords": keywords_list}
    
    # Return as JSON string
    return json.dumps(result_dict, indent=2)

# Example usage
if __name__ == "__main__":
    sample_text = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his leadership of Tesla, SpaceX, and X (formerly Twitter). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk has been considered the wealthiest person in the world since 2021; as of March 2025, Forbes estimates his net worth to be US$345 billion. He was named Time magazine's Person of the Year in 2021.

Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He graduated from the University of Pennsylvania in the U.S. before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became a U.S. citizen.
    """
    
    json_output = extract_keywords_to_json(sample_text)
    print(json_output)