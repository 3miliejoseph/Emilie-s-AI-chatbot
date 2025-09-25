import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI()

# System prompt
system_prompt = """You are an AI assistant for Emilie Joseph's portfolio. You are knowledgeable, friendly, and enthusiastic about Emilie's work.

Emilie is a hard working, multidisciplinary designer. She adapts well and she learns new skills quickly. She focuses on details that non-designers don't even notice. 

Your role is to:
- Provide information about Emilie's skills in UI/UX design, Python, Figma, AI and LLMs, and Adobe Creative Suite
- Answer questions about her 2+ years of UX/UI design and graphic design experience
- Help visitors understand Emilie's expertise in building scalable web applications and highlight mobile and desktop designs
- Maintain a professional yet personable tone
- Emphasize that Emilie is a hard working, multidisciplinary designer
- Highlight that she learns new skills quickly and she focuses on details that non-designers don't  notice
- Guide visitors toward contacting Emilie for potential collaborations
- Remind users that links to Emilie's Github, Linkedin, and Resume are linked in the footer

Featured projects:
- Spotify Redesign: A Smarter Search: A reimagined version of Spotify's search experience to make discovering new tunes faster, clearer, and more intuitive for music lovers.  
- Queuenect: An interactive kiosk that allows people to queue music in public spaces and makes sharing and discovering songs in community spaces fun and intuitive and encourages people to QUEUENECT!   
- Star Wars at UCSD: Designed and iterated a custom website on Figma for UCSD's Star Wars Club, collaborating closely with club leadership to ensure a user-friendly and engaging experience that reflects the club's community and theme.
- The Gallery: Displays recent paintings that Emilie has done 
- Passion projects page that demonstrate creativity and curiosity

If visitors ask about availability or rates, encourage them to reach out directly through the contact form.

Keep responses concise but engaging, maintaining a clean and professional tone without using emojis."""

# Base conversation history
conversation_history = [{"role": "system", "content": system_prompt}]

# Chat function
def chat_with_emilie(user_message, chat_history):
    if not user_message.strip():
        return chat_history, ""
    
    global conversation_history
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})
    
    try:
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            max_tokens=350,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        # Keep conversation history manageable
        if len(conversation_history) > 21:  # 1 system + 20 messages
            conversation_history = [conversation_history[0]] + conversation_history[-20:]
        
        # Add to chat history for display
        chat_history.append([user_message, assistant_message])
        
        return chat_history, ""
        
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        chat_history.append([user_message, error_message])
        return chat_history, ""

# Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;500;600;700&display=swap');

.gradio-container {
    background-color: #EFE6CC !important;
    font-family: 'Urbanist', sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

body, html {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #EFE6CC !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Urbanist', sans-serif !important;
    color: #FF4234 !important;
    margin: 10px 0 !important;
}

button.gr-button,
.gr-button,
button[data-testid*="button"],
.gradio-button,
button {
    background-color: #FF4234 !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: 1px solid #FF4234 !important;
    transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
    font-family: 'Urbanist', sans-serif !important;
}

button.gr-button:hover,
.gr-button:hover,
button[data-testid*="button"]:hover,
.gradio-button:hover,
button:hover {
    background-color: #A61409 !important;
    border-color: #A61409 !important;
    box-shadow: 0 3px 6px rgba(0,0,0,0.12) !important;
}

.gr-textbox textarea {
    font-family: 'Urbanist', sans-serif !important;
    border: 2px solid #FF4234 !important;
    border-radius: 6px !important;
}

.gr-chatbot {
    background-color: #EFE6CC !important;
    border: none !important;
    padding: 0 !important;
    font-family: 'Urbanist', sans-serif !important;
}

.chatbot .message {
    font-family: 'Urbanist', sans-serif !important;
}

footer button, 
footer .gr-button {
    background-color: #d4c5a3 !important;
    border: 1px solid #d4c5a3 !important;
    color: #333 !important;
    font-weight: 500 !important;
}

footer button:hover, 
footer .gr-button:hover {
    background-color: #c5b594 !important;
    border-color: #c5b594 !important;
}
"""

# Gradio app
def create_app():
    with gr.Blocks(title="Emilie's AI Assistant", css=custom_css) as demo:
        gr.Markdown("## Emilie's AI Assistant")
        
        # Initialize chatbot with welcome message
        welcome_message = "Hi there! I'm here to help you learn about Emilie's design work and experience. What would you like to know?"
        chatbot = gr.Chatbot(
            value=[["", welcome_message]], 
            label="",
            height=500
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your question here...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        # Event handlers
        send_btn.click(
            fn=chat_with_emilie, 
            inputs=[msg, chatbot], 
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=chat_with_emilie, 
            inputs=[msg, chatbot], 
            outputs=[chatbot, msg]
        )
    
    return demo

# Create the app
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
