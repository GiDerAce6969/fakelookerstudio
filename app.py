import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="DataPilot AI 🚀",
    page_icon="📊",
    layout="wide"
)

# --- Google Gemini API Configuration ---
# For Streamlit Community Cloud, secrets are stored in st.secrets.
# The key should be named "google_api_key".
try:
    genai.configure(api_key=st.secrets["google_api_key"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-2.0-flash')
except (KeyError, FileNotFoundError):
    st.error("⚠️ **Warning:** Google API key not found.", icon="🚨")
    st.info("Please add your Google API key to your Streamlit secrets. Name it `google_api_key`.")
    st.stop()


# --- Function to Generate AI Response ---
def get_ai_response(df, user_query):
    """
    Generates Python code from a user query using Gemini and executes it.
    """
    # Create a string representation of the DataFrame's head for context
    df_head = df.head().to_string()
    
    prompt = f"""
    You are an expert data analyst AI. You are given a pandas DataFrame named `df`.
    The user wants to perform an analysis on this data.

    Here is the head of the DataFrame to give you context:
    ```
    {df_head}
    ```

    The user's request is: "{user_query}"

    Your task is to generate a single block of Python code to perform the requested analysis.
    - You **MUST** use the pandas DataFrame `df`.
    - For visualizations, you **MUST** use the `plotly.express` library and assign the figure to a variable named `fig`.
    - For data analysis or calculations, the final result should be assigned to a variable named `result`.
    - The code should be a single, runnable block.
    - **DO NOT** include any explanations, comments, or markdown formatting around the code. Only output the raw Python code.

    Example for a plot:
    ```python
    fig = px.bar(df, x='Region', y='Sales', title='Total Sales by Region')
    ```

    Example for a calculation:
    ```python
    result = df['Sales'].sum()
    ```
    
    Now, generate the Python code for the user's request.
    """
    
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        code_to_execute = response.text.strip().replace('```python', '').replace('```', '')
        return execute_code(code_to_execute, df)
    except Exception as e:
        return None, f"An error occurred with the AI model: {e}"


def execute_code(code, df):
    """
    Executes the generated Python code and returns the result or figure.
    """
    try:
        # Create a safe execution environment
        local_vars = {"df": df, "pd": pd, "px": px, "st": st}
        
        # Execute the code
        exec(code, {}, local_vars)
        
        # Retrieve the result
        fig = local_vars.get("fig", None)
        result = local_vars.get("result", None)
        
        return fig, result, code
    except Exception as e:
        return None, f"Error executing code: {e}", code


# --- Streamlit App UI ---

st.title("DataPilot AI 🚀")
st.markdown("Your AI-powered assistant for data analysis and visualization. Upload your data and start asking questions!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# --- Sidebar for Data Upload ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            st.session_state.messages = [] # Clear chat on new upload
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
            
    # Load sample data
    if st.button("Load Sample Data"):
        st.session_state.df = pd.read_csv("sample_data.csv")
        st.success("Sample data loaded!")
        st.session_state.messages = [] # Clear chat


if st.session_state.df is not None:
    with st.sidebar:
        st.header("2. Data Preview")
        st.dataframe(st.session_state.df.head())

    # --- Main Chat Interface ---
    st.header("Chat with Your Data")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message:
                with st.expander("View Generated Code"):
                    st.code(message["code"], language="python")

    # Get user input
    if prompt := st.chat_input("What would you like to analyze? (e.g., 'Plot total sales by region')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("DataPilot is thinking..."):
                fig, result, code = get_ai_response(st.session_state.df, prompt)
                
                ai_response_content = ""
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    ai_response_content = "Here is the visualization you requested."
                elif result is not None:
                    st.write("Here is the result of your analysis:")
                    st.write(result)
                    ai_response_content = "I've completed the calculation for you."
                else:
                    # If there's an error message in the 'result' variable
                    st.error(f"Sorry, I ran into an issue. \n\n{result}")
                    ai_response_content = "I couldn't complete that request. Please try rephrasing or asking something else."
                
                # Add AI response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response_content,
                    "code": code  # Store the code with the message
                })

else:
    st.info("Please upload a CSV or Excel file, or load the sample data to get started.")
