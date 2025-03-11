import streamlit as st
import tensorflow as tf
import tensorflow_text  # Required for tokenization
import os

# Load Tokenizers
def load_tokenizer(filepath):
    with open(filepath, "rb") as f:
        tokenizer = tf.saved_model.load(filepath)
    return tokenizer

# Load Model
@st.cache_resource
def load_model():
    model_path = "model_files/arabic_to_english_transformer_weights.weights"
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load tokenizers
source_tokenizer_path = "model_files/source_tokenizer.subword.subwords"
target_tokenizer_path = "model_files/target_tokenizer.subword.subwords"

source_tokenizer = load_tokenizer(source_tokenizer_path)
target_tokenizer = load_tokenizer(target_tokenizer_path)
model = load_model()

def translate(text):
    # Tokenize input
    input_tensor = source_tokenizer.tokenize([text]).to_tensor()

    # Predict translation
    output_tensor = model(input_tensor)

    # Convert tokens to text
    translated_text = target_tokenizer.detokenize(output_tensor)
    
    return translated_text.numpy()[0].decode("utf-8")

# Streamlit App
st.set_page_config(page_title="Arabic-to-English Translator", layout="centered")

st.title("🌍 Arabic to English Translator")
st.write("Enter Arabic text below and get an English translation.")

# Text input
user_input = st.text_area("📝 Input Arabic Text", height=100)

if st.button("Translate 🚀"):
    if user_input.strip():
        translation = translate(user_input)
        st.success("**Translated Text:**")
        st.write(translation)
    else:
        st.warning("⚠️ Please enter some text before translating.")

