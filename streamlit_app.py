import os
import streamlit as st
import openai 
import pandas as pd
from loguru import logger
import extra_streamlit_components as stx
from PIL import Image
from io import BytesIO
import base64
from pandasai.llm import OpenAI as pandaOpenAI
from pandasai import SmartDatalake
from pandasai.responses.streamlit_response import StreamlitResponse
import docx2txt
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback

st.set_page_config(
    page_title="Skilljourneys ChatGPT",  # Sets the browser tab's title
    page_icon="favicon.ico",        # Sets a browser icon (favicon), here using an emoji
    layout="wide",               # Optional: use "wide" or "centered", the default is "centered"
    initial_sidebar_state="expanded"  # Optional: use "auto", "expanded", or "collapsed"
)

@st.cache_data
def create_model_data_table():
    model_data = {
        "MODEL": [
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106"
        ],
        "DESCRIPTION": [
            "Latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.",
            "GPT-4 Turbo preview model.",
            "GPT-4 Turbo with improved instruction following and JSON mode. Max 4,096 tokens.",
            "Latest version of GPT 4. Continuous upgrades.",
            "Latest version of GPT 4-32K model Continuous upgrades.",
            "Updated GPT-3.5 Turbo model. Fixes text encoding for non-English calls. Max 4,096 tokens.",
            "GPT-3.5 Turbo with improved instruction following. Max 4,096 tokens."
        ],
        "CONTEXT WINDOW": [
            "128,000 tokens",
            "128,000 tokens",
            "128,000 tokens",
            "8,192 tokens",
            "32,768 tokens",
            "16,385 tokens",
            "16,385 tokens"
        ]
    }
    return pd.DataFrame(model_data)

def is_api_key_valid(api_key):
    try:
       client = openai.OpenAI(api_key=api_key)
       client.models.list()
    except openai.AuthenticationError as e:
        return False
    else:
        return True

# Function to get image URL from uploaded file
def get_image_url(image, image_type):
    buffer = BytesIO()
    image.save(buffer, format=image_type)
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    image_url = f"data:image/{image_type};base64,{data_uri}"
    return image_url

# Display the image at the top of the page
st.image("https://lwfiles.mycourse.app/65a6a0bb6e5c564383a8b347-public/05af5b82d40b3f7c2da2b6c56c24bdbc.png", width=500)

# Get OpenAI API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")

# Check if the OpenAI API key is valid
if not openai_api_key:
    st.warning('Please enter your OpenAI API key!', icon='⚠️')
    st.stop();
# check if the API key is not valid
if openai_api_key and not is_api_key_valid(openai_api_key):
    st.warning('Invalid OpenAI API key. Please provide a valid key.', icon='⚠️')
    st.stop()
else:
    client = openai.OpenAI(api_key=openai_api_key) 
    pdclient = pandaOpenAI(openai_api_key)   

chosen_id = stx.tab_bar([
    stx.TabBarItemData(id="1", title="Chat", description=""),
    stx.TabBarItemData(id="2", title="Image Analysis", description=""),
    stx.TabBarItemData(id="3", title="Image Generation", description=""),
    stx.TabBarItemData(id="4", title="Talk with Data (CSV or XLSX)", description=""),
    stx.TabBarItemData(id="5", title="Talk with Documents (PDF, TXT, DOCX)", description="")
])

#----------------------Image processing----------------------
if chosen_id == "2":   
            # Set up the title of the app
    st.title(":rainbow[Skilljourneys Vision GPT]")
    st.subheader(":rainbow[Upload an Image for Analysis]")

    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")

    with st.sidebar:
        uploaded_image = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "gif"])
        model_option = st.selectbox("Select Model", ["gpt-4-vision-preview", "gpt-4-1106-vision-preview"], index=0) 

    user_prompt = st.text_area(label="Enter Prompt", value="What's in this image?")

    if st.button(label="Analyze Image"):
        if openai_api_key and (uploaded_image):
            image_to_analyze = None

            if uploaded_image:
                try:
                    with Image.open(uploaded_image) as image_to_analyze:
                        image_type = image_to_analyze.format
                        # Display and process the uploaded image
                        st.image(image_to_analyze, caption='Uploaded Image',)
                        image_url = get_image_url(image_to_analyze, image_type)
                        # Request to OpenAI
                        response = client.chat.completions.create(
                            model=model_option,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": user_prompt},
                                               {"type": "image_url", "image_url": {"url": image_url}}],
                                }
                            ],
                            max_tokens=300,
                        )
                        # Extract the response text correctly according to the response structure
                        result_text = response.choices[0].message.content
                        st.write(result_text)
                except (FileNotFoundError, IOError, KeyError) as e:
                       st.error(f"Error extracting response: {e}")             
        else:
            st.warning("Please provide an image URL or upload an image and ensure the API key is entered.")
# ----- Image generation -----
elif chosen_id == "3":
        # Set up the title of the app
    st.title(":rainbow[Skilljourneys Vision GPT]")
    st.subheader(":rainbow[Creating images]")
    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")

    with st.sidebar:
       model_option = st.selectbox("Select Model", ["gpt-4-vision-preview", "gpt-4-1106-vision-preview"], index=0)
       # DALL-E settings
       dalle_size = st.selectbox("Select Image Size", ["1024x1024", "512x512", "256x256"], index=0)
       dalle_quality = st.selectbox("Select Image Quality", ["standard", "hd"], index=0)
       dalle_n = st.slider("Number of Images", 1, 4, 1)
    

    dalle_prompt = st.text_area(label="Enter Prompt for DALL-E", value="a beautiful holiday destination")
    if st.button("Generate Image"):
        if openai_api_key:
            try:
                # Request to OpenAI DALL-E
                response = client.images.generate(
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

#----------------------Talking to CSV or Excel----------------------
#https://docs.pandas-ai.com/en/latest/getting-started/
elif chosen_id == "4":  
     # Set up the title of the app
    st.title(":rainbow[Skilljourneys Document GPT]")
    st.subheader(":rainbow[Chat with CSV or XLSX files]")

    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")

    with st.sidebar:
        # Allow user to upload multiple files
        input_files = st.file_uploader(
            "Upload files", type=["xlsx", "csv"], accept_multiple_files=True
        )

    # Main content area
    data_list = []
    # If user uploaded files, load them
    if len(input_files) > 0:
        for input_file in input_files:
            if input_file.name.lower().endswith(".csv"):
                data = pd.read_csv(input_file)
                st.dataframe(data, use_container_width=True)
            else:
                data = pd.read_excel(input_file)
                st.dataframe(data, use_container_width=True)
                data_list.append(data)
        # Create SmartDatalake instance
        df = SmartDatalake(
            data_list,
            config={
                "llm": pdclient,
                "verbose": False,
                "enable_cache":False,
                "save_logs": False,
                "response_parser": StreamlitResponse
            },
        )
        st.subheader("Ask your questions")
        if prompt := st.chat_input("Enter Prompt"):
            result = df.chat(prompt)
            # Display the result and code in two columns
            col1, col2 = st.columns(2)
            with col1:
                # Display the result
                st.header("Answer")
                st.write(result)
            with col2:
                # Display the corresponding code
                st.header("The corresponding code")
                st.code(df.last_code_executed, language="python", line_numbers=True)
#----------------------Talking to PDF, TXT or Word----------------------
elif chosen_id == "5":  
     # Set up the title of the app
    st.title(":rainbow[Skilljourneys Document GPT]")
    st.subheader(":rainbow[Chat with PDF, TXT or DOCX files]")

    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")

    with st.sidebar:
        # Allow user to upload multiple files
        input_files = st.file_uploader(
            "Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True
        )
    if len(input_files) > 0:
    # extract text from uploaded files
        all_text = ""
        for uploaded_file in input_files:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(uploaded_file)
            else:
                st.write(f"Unsupported file type: {uploaded_file.name}")
                continue
            all_text += text
                # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(all_text)
        
            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Update the parameter names

            # show user input
            user_question = st.text_input("Ask a question about the uploaded documents:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
            
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(response)
            
                st.write(response)
# -------------------- Regular Chat API --------------------
else:
    # Set up the title of the app
    st.title(":rainbow[Skilljourneys ChatGPT]")
    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")
    # Allow users to set parameters for the model
    with st.sidebar:
        st.write("Set Model Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0)
        max_tokens = st.slider("Max Tokens", 1, 500, 256)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0)
        frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0)
        presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0)
            # Function to create and cache the model data table
    
    with st.sidebar:
        # Code to display the model data table within an expander
        with st.expander("Model Information"):
            df_models = create_model_data_table()
            st.table(df_models)

    # Allow users to select the model
    model_options = list(df_models["MODEL"])
    # Find the index of 'gpt-3.5-turbo' in the model options list
    default_index = model_options.index('gpt-3.5-turbo') if 'gpt-3.5-turbo' in model_options else 0
    selected_model = st.sidebar.radio("Select the OpenAI model", model_options, index=default_index)

    # Update the model based on user selection
    st.session_state["openai_model"] = selected_model

    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new message
    if prompt := st.chat_input("Enter Prompt"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Getting a response..."):
            try:
                with st.chat_message("assistant"):
                    # Generate the response from the model
                    stream = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        temperature=temperature, 
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stream=True,
                    )
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}", icon='🚨')               
                    
    if st.session_state.messages:
        if st.button('Clear', key="clear"):
            st.session_state.messages = []
            st.rerun()  
