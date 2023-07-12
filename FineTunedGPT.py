#code will run on a url that can be used anywher
#will read pdfs in local 'docs' folder and will first go through that then send to gpt3 16k to do the work
#any request will be more sophisicated and be as current and up to date as the current and up to date the pdfs are

from openai import ChatCompletion
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = ''

# Initialize the messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4-32k", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def search_index(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response if response else None

def chatbot(input_text):
    model = "gpt-3.5-turbo-16k"  # or whatever model you want to use

    # First, try to find an answer in the index.
    index_result = search_index(input_text)

    # If an answer was found in the index, include it in the user's message.
    if index_result is not None:
        input_text += f" I found this in the index: {index_result}"

    # Ask GPT-3.
    # Add the user's message
    messages.append({"role": "user", "content": input_text})
    response = ChatCompletion.create(
      model=model,
      messages=messages
    )
    # Add the model's response
    messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    # Keep only the last 4 messages (2 exchanges)
    messages[:] = messages[-4:]
    return response['choices'][0]['message']['content']

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Lewis Interactive Notes Splunk/Kibana")

# Construct the index
construct_index("docs")

iface.launch(share=True)
