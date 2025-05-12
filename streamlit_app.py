import os
import streamlit as st

import openai 
import pandas as pd
import extra_streamlit_components as stx
from PIL import Image
from io import BytesIO
import base64
from pandasai.llm import OpenAI as pandaOpenAI
import docx2txt
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_openai import OpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import tiktoken
import json


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
            "gpt-4o",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
        ],
        "DESCRIPTION": [
            "Most advanced, multimodal flagship model that’s cheaper and faster than GPT-4 Turbo.",
            "GPT-4",
            "Model with vision capabilities. Vision requests can now use JSON mode and function calling.",
            "GPT-4 Turbo preview model.",
            "The latest GPT-3.5 Turbo model with higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Max 4,096 tokens."
        ],
        "CONTEXT WINDOW": [
            "128,000 tokens",
            "128,000 tokens",
            "128,000 tokens",
            "128,000 tokens",
            "16,385 tokens",

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
    st.title(':rainbow[Welcome to the AI Chat Application!]')
    st.markdown("""
            This application serves as a sandbox for learning and experimenting with artificial intelligence, providing you with a hands-on experience of prompt engineering.

            **Disclaimer:**

            Please note that this application is for training purposes only. The AI-generated responses may contain inaccuracies or reflect unintended biases, as artificial intelligence models can make mistakes.
             
            The responses provided do not necessarily reflect the opinions, beliefs, or views of the company or its affiliates. We encourage you to critically evaluate the output and engage with the material thoughtfully.
            """)
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
        model_option = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo"], index=1) 

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
       model_option = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo"], index=1)
       # DALL-E settings
       dalle_size = st.selectbox("Select Image Size", ["1024x1024", "512x512", "256x256"], index=0)
       dalle_n = st.slider("Number of Images", 1, 4, 1)
    

    dalle_prompt = st.text_area(label="Enter Prompt for DALL-E", value="a beautiful holiday destination")
    if st.button("Generate Image"):
        if openai_api_key:
            try:
                # Request to OpenAI DALL-E
             #   response = client.images.generate(
             #        prompt=dalle_prompt,
             #       size=dalle_size,
             #       n=dalle_n,
             #   )
                response = client.images.generate(
                    model="dall-e-3",  # critical for realism
                    prompt=dalle_prompt,
                    size="1024x1024",
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
        input_file = st.file_uploader(
            "Upload files", type=["xlsx", "csv"], accept_multiple_files=False
        )
    with st.sidebar:
        st.write("Set Model Parameters")
        model_option = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
        

    # Display chat history
    for message in st.session_state['messages']:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    # If user uploaded files, load them
    if input_file is not None:

        if input_file.name.lower().endswith(".csv"):
            data = pd.read_csv(input_file)
            st.dataframe(data, use_container_width=True)
        else:
            data = pd.read_excel(input_file)
            st.dataframe(data, use_container_width=True)

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=temperature, model=model_option, openai_api_key=openai_api_key),
            data,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True 
        )
 
        st.subheader("Ask your questions")
        if prompt := st.chat_input("Enter Prompt"):
            try:
                json_prompt = json.dumps(prompt)
                result = agent.invoke(json_prompt)
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Display the parsed result
                with st.chat_message("assistant"):
                    if isinstance(result, dict):
                        response_content = result.get('output','No output property found')
                        st.markdown(response_content)
                    else:
                        st.write(result)
            except (TypeError, ValueError) as e:
                st.error(f"Failed to convert data: {e}")
    

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
    with st.sidebar:
        st.write("Set Model Parameters")
        model_option = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo","gpt-3.5-turbo"], index=2)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5)  

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
            embeddings = HuggingFaceEmbeddings()
            knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Update the parameter names

            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = model_option,temperature=temperature)
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=knowledge_base.as_retriever(), 
                memory=memory
            )
            st.session_state.conversation = conversation_chain

            # show user input
            user_question = st.chat_input("Ask a question about the uploaded documents:")
            if user_question:
                with get_openai_callback() as cb:
                    response = st.session_state.conversation({'question':user_question})
                st.session_state.chat_history = response['chat_history']

                response_container = st.container()

                with response_container:
                    for i, messages in enumerate(st.session_state.chat_history):
                        if i % 2 == 0:
                            message(messages.content, is_user=True, key=str(i))
                        else:
                            message(messages.content, key=str(i))
                    st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")


# -------------------- Regular Chat API --------------------
else:
    # Set up the title of the app
    st.title(":rainbow[Skilljourneys ChatGPT]")
    # Link to Trivera Tech website
    st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")
    # Allow users to set parameters for the model
    #What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
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
    # If there is at least one message in the chat, we display the options
    if len(st.session_state["messages"]) > 0:

        # We set the space between the icons thanks to a share of 100
        cols_dimensions = [10,30,30,30]
        col0, col1, col2, col3 = st.columns(cols_dimensions)

        with col1:
            # Converts the list of messages into a JSON Bytes format
            json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")
            # And the corresponding Download button
            st.download_button(
                label="📥 Save chat!",
                data=json_messages,
                file_name="chat_conversation.json",
                mime="application/json",
            )
        with col2:
            if st.session_state.messages:
                if st.button('Clear Chat 🧹', key="clear"):
                    st.session_state.messages = []
                    st.rerun()  
        with col3:
            # We initiate a tokenizer
            enc = tiktoken.get_encoding("cl100k_base")
            # We encode the messages
            tokenized_full_text = enc.encode(
                " ".join([item["content"] for item in st.session_state["messages"]])
            )
            # And display the corresponding number of tokens
            st.markdown(f"💬 {len(tokenized_full_text)} tokens")

