import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
from functools import lru_cache


@lru_cache(maxsize=1)
def load_led_model(model_name="allenai/led-base-16384"):
    """
    Caches and loads the LED tokenizer and model.
    Only loads once and reuses on subsequent calls.
    """
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def generate_highlights(text, max_input_length=16000, max_output_length=256):
    """
    Generates highlights from long-form input text using LED model.
    """
    tokenizer, model = load_led_model()

    # Tokenize input with truncation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    # Assign global attention to the first token
    global_attention_mask = torch.zeros_like(inputs['input_ids'])
    global_attention_mask[:, 0] = 1

    # Generate summary/highlight
    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        global_attention_mask=global_attention_mask,
        max_length=max_output_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def main():
    # Load input text
    with open("sample_long_document.txt", "r", encoding="utf-8") as f:
        long_text = f.read()

    # Generate highlight
    highlight = generate_highlights(long_text)

    print("\n🟨 Extracted Highlight:\n")
    print(highlight)


if __name__ == "__main__":
    main()
