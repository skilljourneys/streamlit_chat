import streamlit as st
import openai
from PIL import Image
import requests
from io import BytesIO
import base64

# Function to get image URL from uploaded file
def get_image_url(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    return "data:image/jpeg;base64," + data_uri

st.set_page_config(
    page_title="Skilljourneys Vision GPT",  # Sets the browser tab's title
    page_icon="favicon.ico",        # Sets a browser icon (favicon), here using an emoji
    layout="wide",               # Optional: use "wide" or "centered", the default is "centered"
    initial_sidebar_state="expanded"  # Optional: use "auto", "expanded", or "collapsed"
)

# Display the image at the top of the page
st.image("https://lwfiles.mycourse.app/65a6a0bb6e5c564383a8b347-public/05af5b82d40b3f7c2da2b6c56c24bdbc.png", width=500)
# Set up the title of the app
st.title("Skilljourneys Vision GPT")


# Sidebar for API key and options
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
# Check if the OpenAI API key is valid
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠️')

model_option = st.sidebar.selectbox("Select Model", ["gpt-4-vision-preview", "gpt-4-1106-vision-preview"], index=0)

# Check for client initialization before displaying the input field
if locals() or 'client' in globals():  
    # Tab layout
    tab1, tab2 = st.tabs(["Image Analysis", "Image Generation"])

    with tab1:
        st.subheader("Upload an Image or Provide Image URL for Analysis")

        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf", "docx"])
        user_prompt = st.text_area("Enter Prompt", "What's is this resource?")

        if st.button("Analyze Image"):
            if openai_api_key and (uploaded_file):
                image_to_analyze = None

                if uploaded_file and uploaded_file.type in ("text/plain", "application/octet-stream"):
                    file_content = base64.b64encode(uploaded_file.read()).decode('utf-8')  
                    # Configure OpenAI client
                    openai.api_key = openai_api_key
                    prompt = f"Considering the uploaded content: {file_content}\n {user_prompt}"
                    st.write(prompt)
                    # Request to OpenAI
                    response = openai.ChatCompletion.create(
                        model=model_option,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                        max_tokens=300,
                    )
                    try:
                        # Extract the response text correctly according to the response structure
                        result_text = response['choices'][0]['message']['content']
                        st.write(result_text)
                    except KeyError as e:
                        st.error(f"Error extracting response: {e}")
                        st.write(response)  # Print the whole response for debugging
            else:
                st.warning("Please provide an image URL or upload an image and ensure the API key is entered.")

    # Adding the second tab for DALL-E image generation
    with tab2:
        st.subheader("Generate Images")

        # DALL-E settings
        dalle_model = st.selectbox("Select DALL-E Model", ["gpt-4","dall-e-3", "dall-e-2"], index=0)
        dalle_prompt = st.text_input("Enter Prompt for DALL-E", "a white siamese cat")
        dalle_size = st.selectbox("Select Image Size", ["1024x1024", "512x512", "256x256"], index=0)
        dalle_quality = st.selectbox("Select Image Quality", ["standard", "hd"], index=0)
        dalle_n = st.slider("Number of Images", 1, 4, 1)

        if st.button("Generate Image"):
            if openai_api_key:
                # Configure OpenAI client
                openai.api_key = openai_api_key

                try:
                    # Request to OpenAI DALL-E
                    response = openai.Image.create(
                        model=dalle_model,
                        prompt=dalle_prompt,
                        size=dalle_size,
                        quality=dalle_quality,
                        n=dalle_n,
                    )

                    # Display generated images
                    for i in range(len(response.data)):
                        st.image(response.data[i].url, caption=f"Generated Image {i+1}")

                except Exception as e:
                    st.error(f"Error generating images with DALL-E: {e}")
            else:
                st.warning("Please enter the OpenAI API key.")