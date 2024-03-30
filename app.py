import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_text(input_text, max_length=32, num_beams=5, do_sample=False, no_repeat_ngram_size=2):
    """
    Generate text based on the given input text.
    Parameters:
    - input_text (str): The input text to start generation from.
    - max_length (int): Maximum length of the generated text.
    - num_beams (int): Number of beams for beam search.
    - do_sample (bool): Whether to use sampling or not.
    - no_repeat_ngram_size (int): Size of the n-gram to avoid repetition.
    Returns:
    - generated_text (str): The generated text.
    """
    # Encode the input text and move it to the appropriate device
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True)['input_ids']
    # Generate text using the model
    output = model.generate(input_ids, max_length=max_length, num_beams=num_beams,
                            do_sample=do_sample, no_repeat_ngram_size=no_repeat_ngram_size)
    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Create Gradio interface
input_text = gr.Textbox(lines=10, label="Input Text", placeholder="Enter text for text generation...")
output_text = gr.Textbox(label="Generated Text")

gr.Interface(generate_text, input_text, output_text,
             title="Text Generation with GPT-2",
             description="Generate text using the GPT-2 model.",
             theme="default", 
             allow_flagging="never").launch()
