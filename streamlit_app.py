import streamlit as st
import openai 
import pandas as pd

st.set_page_config(
    page_title="Skilljourneys ChatGPT",  # Sets the browser tab's title
    page_icon="favicon.ico",        # Sets a browser icon (favicon), here using an emoji
    layout="wide",               # Optional: use "wide" or "centered", the default is "centered"
    initial_sidebar_state="expanded"  # Optional: use "auto", "expanded", or "collapsed"
)

# Display the image at the top of the page
st.image("https://lwfiles.mycourse.app/65a6a0bb6e5c564383a8b347-public/05af5b82d40b3f7c2da2b6c56c24bdbc.png", width=500)
# Set up the title of the app
st.title(":rainbow[Skilljourneys ChatGPT]")
# Link to Trivera Tech website
st.markdown(":blue[For more information, visit [Triveratech](https://www.triveratech.com).]")

# Get OpenAI API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")

# Check if the OpenAI API key is valid
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠️')
else:
    # Initialize the OpenAI client with the valid API key
    client = openai.OpenAI(api_key=openai_api_key)

# Allow users to set parameters for the model
with st.sidebar:
    st.write("Set Model Parameters")
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0)
    max_tokens = st.slider("Max Tokens", 1, 500, 256)
    top_p = st.slider("Top P", 0.0, 1.0, 1.0)
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0)
    presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check for client initialization before displaying the input field
if locals() or 'client' in globals():  
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

# Function to create and cache the model data table
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
        ],
        "TRAINING DATA": [
            "Up to Dec 2023",
            "Up to Dec 2023",
            "Up to Apr 2023",
            "Up to Sep 2021",
            "Up to Sep 2021",
            "Up to Sep 2021",
            "Up to Sep 2021"
        ]
    }
    return pd.DataFrame(model_data)

with st.sidebar:
    # Code to display the model data table within an expander
    with st.expander("Model Information"):
        df_models = create_model_data_table()
        st.table(df_models)

    # Data policy and model endpoint compatibility expander
    with st.expander("OpenAI API Data Usage Policy"):
        st.markdown("""
        **How we use your data**
        Your data is your data.

        As of March 1, 2023, data sent to the OpenAI API will not be used to train or improve OpenAI models (unless you explicitly opt in). One advantage to opting in is that the models may get better at your use case over time.

        To help identify abuse, API data may be retained for up to 30 days, after which it will be deleted (unless otherwise required by law). For trusted customers with sensitive applications, zero data retention may be available. With zero data retention, request and response bodies are not persisted to any logging mechanism and exist only in memory in order to serve the request.

        Note that this data policy does not apply to OpenAI's non-API consumer services like ChatGPT or DALL·E Labs.

        **Default usage policies by endpoint**

        | ENDPOINT | DATA USED FOR TRAINING | DEFAULT RETENTION | ELIGIBLE FOR ZERO RETENTION |
        | --- | --- | --- | --- |
        | /v1/chat/completions | No | 30 days | Yes, except image inputs |
        | /v1/files | No | Until deleted by customer | No |
        | ... | ... | ... | ... |
        | /v1/completions | No | 30 days | Yes |
        * Image inputs via the gpt-4-vision-preview model are not eligible for zero retention.

        * For the Assistants API, we are still evaluating the default retention period during the Beta. We expect that the default retention period will be stable after the end of the Beta.

        **Model endpoint compatibility**

        | ENDPOINT | LATEST MODELS |
        | --- | --- |
        | /v1/assistants | All models except gpt-3.5-turbo-0301 supported. |
        | ... | ... |
        This list excludes all of our deprecated models.

        For details, see our API data usage policies. To learn more about zero retention, get in touch with our sales team.
        """, unsafe_allow_html=True)



# Allow users to select the model
model_options = list(df_models["MODEL"])
# Find the index of 'gpt-3.5-turbo' in the model options list
default_index = model_options.index('gpt-3.5-turbo') if 'gpt-3.5-turbo' in model_options else 0
selected_model = st.sidebar.radio("Select the OpenAI model", model_options, index=default_index)

# Update the model based on user selection
st.session_state["openai_model"] = selected_model