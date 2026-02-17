from deepface import DeepFace
import os
import cv2
import google.generativeai as genai
from PIL import Image
import streamlit as st
import tempfile

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyA7dPzirWJrcxYaVtk8lP4RrGKnXhtIpcM"
genai.configure(api_key=GEMINI_API_KEY)

st.title("Sebastião")

# session start
if "name" not in st.session_state:
    st.session_state.name = None
if "baseline_path" not in st.session_state:
    st.session_state.baseline_path = None
if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# user's name
if st.session_state.name is None:
    name = st.text_input("Your name:")
    if name:
        st.session_state.name = name
        st.rerun()
    st.stop()

# Capture baseline image (only once)
if st.session_state.baseline_path is None:
    st.write(f"Welcome, {st.session_state.name}! Please upload a baseline photo.")
    uploaded_file = st.file_uploader(
        "Choose an image of yourself",
        type=["jpg", "jpeg", "png"],
        key="baseline_uploader"
    )
    if uploaded_file is not None:
        # Save baseline image permanently for the session
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state.baseline_path = tmp.name
        st.success("Baseline image saved!")
        st.rerun()
    st.stop()

# chat interface
st.write(f"Chatting as {st.session_state.name}")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user thing
prompt = st.chat_input(
    "Type your message (attach an image for verification)",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

if prompt and prompt.files:
    # User has submitted at least an image
    user_text = prompt.text if prompt.text else ""
    image_file = prompt.files[0]

    # saveing the image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.getvalue())
        current_image_path = tmp.name

    try:
        # Verify face against baseline
        if not os.path.exists(st.session_state.baseline_path):
            st.error("Baseline image not found. Please restart.")
            st.stop()

        # face verification
        result = DeepFace.verify(
            img1_path=st.session_state.baseline_path,
            img2_path=current_image_path,
            enforce_detection=False  # Avoid crashes if no face detected
        )
        is_verified = result["verified"]
        if is_verified:
            st.success(f"✅ Face verified as {st.session_state.name}")
        else:
            st.error("❌ Face does not match baseline. Message ignored.")
            st.stop()  # Stop processing this input

        # Emotion analysis
        st.write("Analyzing emotion...")
        analysis = DeepFace.analyze(
            img_path=current_image_path,
            actions=['emotion'],
            enforce_detection=False
        )
        emotion = analysis[0]['dominant_emotion']
        st.info(f"Detected emotion: {emotion}")

        # LLM template for Gemini
        template = f"""
        You are Sebastião, a personal AI of {st.session_state.name}.
        Respond naturally to him.
        He seems {emotion} right now.

        Context of previous conversation:
        {st.session_state.context}

        His current message:
        {user_text}

        Please respond as Sebastião in a helpful, personal way.
        """

        st.write("Sebastião is thinking...")

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-3-flash-preview')  # You can also use 'gemini-1.5-pro'

        # Prepare the message with image if available
        if os.path.exists(current_image_path):
            # Upload image to Gemini
            image_file_for_gemini = genai.upload_file(current_image_path)

            # Generate response with image context
            response = model.generate_content([
                template,
                image_file_for_gemini
            ])
        else:
            # Generate response without image
            response = model.generate_content(template)

        assistant_reply = response.text

        # store conversation
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.session_state.context += f"User: {user_text}\nSebastião: {assistant_reply}\n"

        # display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback

        st.error(traceback.format_exc())  # More detailed error info
    finally:
        # clean up the temporary image file
        if os.path.exists(current_image_path):
            os.unlink(current_image_path)

elif prompt and not prompt.files:
    st.warning("Please attach an image for face verification.")
