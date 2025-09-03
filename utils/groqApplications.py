import os
import json
from groq import Groq
import requests
import io
import re
import sys
from dotenv import load_dotenv

load_dotenv()
# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles special types from the Groq API."""
    def default(self, obj):
        # Convert to dictionary if the object has a __dict__ attribute
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Convert to string for other non-serializable objects
        try:
            return str(obj)
        except:
            return super().default(obj)

def transcribe_audio_from_url_groq(url, model="whisper-large-v3-turbo", prompt=None, 
                                  response_format="verbose_json", timestamp_granularities=None, 
                                  language=None, temperature=0.0):
    print(f"Transcribing audio from URL: {url}")
    """
    Transcribe an audio file from a URL using Groq's audio transcription API.
    
    Args:
        url (str): URL of the audio file to transcribe
        model (str): Model to use for transcription
        prompt (str, optional): Context or spelling hints for the transcription
        response_format (str, optional): Format of the response
        timestamp_granularities (list, optional): List of timestamp granularities
        language (str, optional): Language code of the audio
        temperature (float, optional): Temperature for the model
        
    Returns:
        dict: Transcription result from Groq
    """
    # Download the content from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch audio from URL: Status code {response.status_code}")
    
    # Create a file-like object from the content
    audio_data = io.BytesIO(response.content)
    audio_data.name = 'audio.mp3'  # Set a filename for the file-like object
    
    # Prepare parameters for the API call
    params = {
        "file": audio_data,
        "model": model,
        "temperature": temperature
    }
    
    # Add optional parameters if provided
    if prompt:
        params["prompt"] = prompt
    if response_format:
        params["response_format"] = response_format
    if timestamp_granularities:
        params["timestamp_granularities"] = timestamp_granularities
    if language:
        params["language"] = language
    
    try:
        # Create a transcription
        transcription = client.audio.transcriptions.create(**params)
        
        # Extract relevant attributes from the Transcription object
        transcription_dict = {}
        
        # First, try to get attributes directly
        if hasattr(transcription, 'text'):
            transcription_dict['text'] = transcription.text
        
        if hasattr(transcription, 'segments'):
            segments_list = []
            for segment in transcription.segments:
                segment_dict = {}
                # Extract common segment attributes
                for attr in ['id', 'seek', 'start', 'end', 'text', 'tokens', 
                             'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob']:
                    if hasattr(segment, attr):
                        segment_dict[attr] = getattr(segment, attr)
                # segments_list.append(segment_dict)
            # transcription_dict['segments'] = segments_list
        
        if hasattr(transcription, 'words'):
            words_list = []
            for word in transcription.words:
                word_dict = {}
                # Extract common word attributes
                for attr in ['word', 'start', 'end']:
                    if hasattr(word, attr):
                        word_dict[attr] = getattr(word, attr)
                words_list.append(word_dict)
            # transcription_dict['words'] = words_list
        
        # Additional metadata
        for attr in ['task', 'language', 'duration']:
            if hasattr(transcription, attr):
                transcription_dict[attr] = getattr(transcription, attr)
        transcription_dict = json.dumps(transcription_dict, indent=2, cls=CustomJSONEncoder)
        return transcription_dict
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def LLMChat(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the following question: {query}",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content
# Example usage:
if __name__ == "__main__":
    url = 'https://tmpfiles.org/dl/28918378/sampleordertakingcustomersupportphilippines.mp3'
    
    try:
        result = transcribe_audio_from_url_groq(
            url=url,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
            temperature=0.0
        )
        
        # Pretty print the result
        print(result)
        
        # Save the result to a file
        # with open('transcription_result.json', 'w') as f:
        #     json.dump(result, f, indent=2, cls=CustomJSONEncoder)
        # print("\nTranscription saved to transcription_result.json")
        
    except Exception as e:
        print(f"Error: {e}")