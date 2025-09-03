from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cache dictionary to avoid reloading
_model_cache = {
    "tokenizer": None,
    "model": None
}

def load_gemma_model():
    """
    Loads and caches the Gemma model and tokenizer.
    Returns:
        tokenizer, model
    """
    global _model_cache
    if _model_cache["tokenizer"] is None or _model_cache["model"] is None:
        print("Loading Gemma model and tokenizer...")
        _model_cache["tokenizer"] = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        _model_cache["model"] = AutoModelForCausalLM.from_pretrained(
            "google/gemma-7b-it",
            torch_dtype=torch.bfloat16
        )
    return _model_cache["tokenizer"], _model_cache["model"]

def generate_text(prompt: str) -> str:
    """
    Generates text from the given prompt using cached Gemma model.
    
    Args:
        prompt (str): Input prompt text.
        
    Returns:
        str: Generated response.
    """
    tokenizer, model = load_gemma_model()
    input_ids = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "pls etract key phrases from the following text : Thank you for calling Martha's Flores. How may I assist you? Hello, I'd like to order flowers and I think you have what I'm looking for. I'd be happy to take care of your order. May I have your name please? Randall Thomas. Randall Thomas. Can you spell that for me? Randall, R-A-N-D-A-L-L, Thomas, T-H-O-N-A-N. Thank you for that information Randall. May I have your home or office number or area code first? Air code 409, then 866-5088. That's 409-866-5088. Do you have a fax number or email address? My email is randall.thomas at gmail.com. randall.thomas at gmail.com. May have your shipping address. 6800. Okay. Gladys Avenue, Beaumont, Texas, zip code 77706. Gladys Avenue, Beaumont, Texas, zip code 77706. Thank you for the information. What products were you interested in purchasing? Red Roses. Probably a dozen. One dozen of Red Roses. Do you want long stems? Sure. Alright. Randall, let me process your order. One moment, please. Okay. Okay. Randall, you're ordering one dozen long-stent red roses. The total amount of your order is $40, and it will be shipped to your address within 24 hours. I was looking to deliver my roses again. Within 24 hours. Okay, no problem. Is there anything else I can help you with? That's all for now, thanks. No problem, Randall. Thank you for calling Martha's Florist. Have a nice day. Thank you."
    print(generate_text(prompt))
