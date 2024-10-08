import gradio as gr
from openai import OpenAI
import os
from prompts import *
from pinecone_index import qa
from gtts import gTTS
import ffmpeg

client = OpenAI(api_key="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")

conversation_history = []


def transcribe(audio_rec):
    
        audio_file = open(audio_rec, "rb")
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language="en", response_format="text")
        
    
        return transcript_response
  
    
def chatcompletion(transcribed_text):    
    global conversation_history

    conversation_history.append({"role": "user", "content": transcribed_text})

    message = [system_prompt] + conversation_history

    vectorstore_response = qa.invoke({"query": transcribed_text, "system": message})
    
    conversation_history.append({"role": "assistant", "content":vectorstore_response.get('result')})

    audioobj = gTTS(text = vectorstore_response.get('result') , 
                    lang = 'en', 
                    slow = False)
    
    audioobj.save("Temp.mp3")
   
    return [transcribed_text,vectorstore_response.get('result'),'Temp.mp3']
    


def process_audio_to_chat(audio):
   
    transcribed_text = transcribe(audio)
    
    
    ai_response = chatcompletion(transcribed_text)
    
    
    return  ai_response

def reset_conversation():
    global conversation_history
    conversation_history = []
    return "Conversation history reset."


audio_input=gr.Audio(type="filepath")
output_1 = gr.Textbox(label="Query")
output_2 = gr.Textbox(label="Product Details")
output_3 = gr.Audio("Temp.mp3")
#Gradio interface
iface = gr.Interface(
    fn=process_audio_to_chat,                 # Function to call
    inputs = audio_input,                     # Input from microphone                     
    outputs=[
        output_1,  output_2, output_3
    ],                                        # Output type
    title="Sales Assistant",                  # Title of the interface
    description="Ask anything ",              # Description for the chatbot

)

iface.launch(share=True)

