import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import pipeline
from scraping import extract_comment, extract_video_id
import tensorflow as tf
import numpy as np
import re
import requests
from PIL import Image
from io import BytesIO
from keras.models import load_model
import cv2
from ultralytics import YOLO
from OCR_Project.ocr.PaddleOCR import app
import chatgpt as cgpt

model_cat_vs_dog = load_model("CatVsDogModel50.h5")
yolo_model = YOLO("best (8).pt")




def classify_cat_vs_dog(image):
    try:
        # Convert the image to a format suitable for prediction
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img_inp = img.reshape((1, 128, 128, 3))

        # Make a prediction
        predictions = model_cat_vs_dog.predict(img_inp)

        # Determine the class label
        if predictions[0][0] >= 0.5:
            return "Dog"
        else:
            return "Cat"
    except ValueError:
        st.error("Error processing the image. Please ensure it is a valid image.")
        return None

def object_detection_app(model, uploaded_image):
    # Add a slider for adjusting the confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    if uploaded_image:
        # Save the uploaded image to a temporary file
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.read())

        # Read the uploaded image using cv2
        inp_img = cv2.imread(temp_file_path)

        # Perform prediction with the specified confidence threshold
        results = model.predict(source=inp_img, conf=confidence_threshold)[0]
        print(results)

        # Convert BGR to RGB
        inp_img_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes and labels on the image
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > confidence_threshold:
                cv2.rectangle(inp_img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                label = f"{results.names[int(class_id)].upper()}rs ({score:.2f})"
                cv2.putText(inp_img_rgb, label, (int(x1), int(y1 + 50)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the processed image
        st.image(inp_img_rgb, caption="Processed Image", use_column_width=True)

        # Save the processed image
        cv2.imwrite("out.jpg", inp_img_rgb)


def currency_det(): 
    # Load the YOLO model
    yolo_model = YOLO("best (8).pt")

    # Create a Streamlit app
    st.title("YOLO Object Detection")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Call the object detection function directly
    if uploaded_image:
        object_detection_app(yolo_model, uploaded_image)
 
def main():
    st.title("🚀✨ Computer Vision Sentix ") 
    # Create tabs
    selected_tab = st.sidebar.selectbox("Select an analysis task", ["Sentiment Analysis", "Youtube Comments Analysis","Currency Detection","Optical Character Recognition"])

    if selected_tab == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_tab == "Youtube Comments Analysis":
        youtube_comments_analysis()
    # elif selected_tab == "Cat vs Dog Classification":
    #     cat_vs_dog_classification()
    elif selected_tab == "Currency Detection":
        currency_det()
    elif selected_tab=="Optical Character Recognition":
        st.subheader("📈 OPTICAL CHARACTER RECOGNITION ")
        app.ocr_prediction()
        cgpt.chatting()

        
        

def clean_text(text):
   
    text = re.sub(r'[\U0001F600-\U0001F6FF]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+k\scomments|\d+milion\slikes|\d+b\sviews', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+:\d+\s(am|pm)|\d+/\d+/\d+\sح/س|master\s?piece|for\swhat\s?exactly', '', text, flags=re.IGNORECASE)
    return text

def predict_sent(text):
   
    MODEL = "My_Sent_Model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    text = clean_text(text)
    res = {}
    max_chunk_size = 250
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    sentiment_scores = []
    for chunk in chunks:
        encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        output = model(**encoded_input)
        scores = output.logits.softmax(dim=-1).detach().numpy()
        sentiment_scores.extend(scores)
    average_scores = np.mean(sentiment_scores, axis=0)
    sorted_scores = sorted([(label, score) for label, score in zip(config.id2label.values(), average_scores)], key=lambda x: x[1], reverse=True)
    for i, (label, score) in enumerate(sorted_scores):
        res[label] = np.round(float(score * 100), 2)
        print(f"{i+1}) {label}: {np.round(score*100, 2)}")
    return res

def predict_emotion(text):
    
    emotion_classifier = pipeline("text-classification", model="Emotion_Model", return_all_scores=True)
    emotion_results = emotion_classifier(clean_text(text))
    sorted_results = sorted(emotion_results[0], key=lambda x: x['score'], reverse=True)
    return sorted_results

def predict_comments(url):
   
    id = extract_video_id(url)
    if id is not None:
        ls = extract_comment(id)
        text = " ".join(ls)
        return text
    else:
        return ""

def sentiment_analysis():
    st.subheader("📈 Sentiment Analysis")
    st.write("Perform sentiment analysis on text. Enter text in the text area and click 'Predict Sentiment' to see the analysis results.")
    
    user_input = st.text_area("Enter your text:")
    
    if st.button("Predict Sentiment"):
        cleaned_input = clean_text(user_input)
        st.write("Analyzing....")
        
        with st.spinner():
            sentiment_scores = predict_sent(cleaned_input)
            st.success("Analysis complete!")

        st.write("Sentiment Analysis Results:")
        positive_percentage = float(sentiment_scores["positive"])
        negative_percentage = float(sentiment_scores["negative"])
        neutral_percentage = float(sentiment_scores["neutral"])
        st.write(f"Positive: {positive_percentage:.2f}%")
        st.progress(positive_percentage / 100)
        st.write(f"Neutral: {neutral_percentage:.2f}%")
        st.progress(neutral_percentage / 100)
        st.write(f"Negative: {negative_percentage:.2f}%")
        st.progress(negative_percentage / 100)

def youtube_comments_analysis():
    st.subheader("📺 Youtube Comments Analysis")
    st.write("Analyze sentiments of comments on a YouTube video. Enter the YouTube video URL and click 'Predict Sentiment' to see the analysis results.")
    
    user_input = st.text_area("Enter your url:")

    if st.button("Predict Sentiment"):
        text = predict_comments(user_input)
        if text:
            st.write("Fetching comments...")
            with st.spinner():
                sentiment_scores = predict_sent(clean_text(text))
                st.success("Analysis complete!")

            st.write("Sentiment Analysis Results:")
            positive_percentage = float(sentiment_scores["positive"])
            negative_percentage = float(sentiment_scores["negative"])
            neutral_percentage = float(sentiment_scores["neutral"])
            st.write(f"Positive: {positive_percentage:.2f}%")
            st.progress(positive_percentage / 100)
            st.write(f"Neutral: {neutral_percentage:.2f}%")
            st.progress(neutral_percentage / 100)
            st.write(f"Negative: {negative_percentage:.2f}%")
            st.progress(negative_percentage / 100)
        else:
            st.error("Please paste a valid YouTube video URL excluding shorts and live streams.")

# def cat_vs_dog_classification():
#     st.subheader("🐱🐶 Cat vs. Dog Classification")
#     st.write("Choose an option to classify an image as cat or dog.")
#     # Radio button to choose between pasting URL or uploading image
#     option = st.radio("Choose an option:", ("Paste an image URL", "Upload an image"))

#     if option == "Paste an image URL":
#         url = st.text_input("Paste the image URL:")
#         if url:
#             try:
#                 response = requests.get(url)
#                 image = Image.open(BytesIO(response.content))
#             except Exception as e:
#                 st.error("Error downloading the image. Please check the URL.")
#                 image = None
#     elif option == "Upload an image":
#         try:
#             uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
#             if uploaded_image:
#                 image = Image.open(uploaded_image)
#         except :
#             st.error("Invalid Image")

#     if "image" in locals() and image is not None:
#         st.image(image, caption="Image", use_column_width=True)

#         if st.button("Classify"):
#             with st.spinner("Classifying..."):
#                 result = classify_cat_vs_dog(image)

#             st.subheader("Classification Result:")
#             st.subheader(result)


if __name__ == "__main__":
    main()