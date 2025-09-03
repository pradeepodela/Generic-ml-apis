from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

def ollamaParserClient(text, template , model):
    response = client.chat(model=model, messages=[
        {
            'role': 'user',
            'content': f'Convert the following unstructured text into : {template} in the followin json format make sure the generated json is highly accurate and is highly accurate \n\n{text}',
        },
    ])
    return response['message']['content']