import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crop_model_small.keras", compile=False)
    names = json.load(open("class_names.json"))
    return model, names

model, class_names = load_model()

def nice(name):
    return name.replace('___', ' - ').replace('_', ' ').replace('  ', ' ').strip()

st.title("🌿 Crop Disease Detection")
st.write("Upload a photo of a single leaf (apple, bell pepper, blueberry, cherry, corn, grape, "
         "orange, peach, potato, raspberry, soybean, squash, strawberry, tomato).")

file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img)

    with st.spinner("Checking the leaf..."):
        x = tf.image.resize(np.array(img, dtype=np.float32), (224, 224))
        probs = model.predict(x[tf.newaxis, ...], verbose=0)[0]

    top3 = np.argsort(probs)[::-1][:3]
    best = top3[0]

    if probs[best] < 0.60:
        st.warning(f"⚠️ Not sure ({probs[best]*100:.0f}%). Please retake the photo: "
                   "one leaf filling most of the photo, daylight, spots in focus.")
    else:
        st.success(f"✅ {nice(class_names[best])} — Confidence: {probs[best]*100:.0f}%")

    st.subheader("Top 3 guesses")
    for i in top3:
        st.write(f"{nice(class_names[i])}: {probs[i]*100:.0f}%")
        st.progress(float(probs[i]))
