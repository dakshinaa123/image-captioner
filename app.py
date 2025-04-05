# 📦 Imports
import gradio as gr
import torch
import requests
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# 🖼️ Caption Model: ViT-GPT2
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# 🌍 Translation Model: M2M100
translation_model_name = "facebook/m2m100_418M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# 🌐 Language Mapping
lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn"
}

# 🧠 Generate Caption (English)
def generate_caption(image):
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=40, num_beams=4)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# 🌐 Translate Caption
def translate_text(text, target_lang):
    target_lang_code = lang_code_map.get(target_lang, "hi")
    translation_tokenizer.src_lang = "en"
    encoded = translation_tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code)
    )
    return translation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# 📁 From Upload
def handle_upload(image, lang):
    caption = generate_caption(image)
    translated = translate_text(caption, lang)
    return caption, translated

# 🌐 From URL
def handle_url(url, lang):
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    caption = generate_caption(image)
    translated = translate_text(caption, lang)
    return caption, translated

# 🎛️ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("##  Multilingual Image Caption Generator")

    with gr.Tab("📁 Upload Image"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            lang_choice = gr.Dropdown(choices=list(lang_code_map.keys()), value="Hindi", label="Target Language")
        eng_out = gr.Textbox(label="📝 English Caption")
        trans_out = gr.Textbox(label="🌍 Translated Caption")
        btn = gr.Button("🚀 Generate & Translate")
        btn.click(fn=handle_upload, inputs=[image_input, lang_choice], outputs=[eng_out, trans_out])

    with gr.Tab("🌐 From URL"):
        with gr.Row():
            url_input = gr.Textbox(label="Paste Image URL")
            lang_choice2 = gr.Dropdown(choices=list(lang_code_map.keys()), value="Hindi", label="Target Language")
        eng_out2 = gr.Textbox(label="📝 English Caption")
        trans_out2 = gr.Textbox(label="🌍 Translated Caption")
        btn2 = gr.Button("🚀 Generate from URL")
        btn2.click(fn=handle_url, inputs=[url_input, lang_choice2], outputs=[eng_out2, trans_out2])

demo.launch(share=True)
