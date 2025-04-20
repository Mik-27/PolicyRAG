from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

from application import PolicyRAG

app = Flask(__name__)
socketio = SocketIO(app)

rag = PolicyRAG()

@app.route('/chat')
def chat():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(data):
    start = time.time()
    user_message = data['message']
    
    # Process the user message
    response = generate_chatbot_response(user_message)
    end = time.time()
    print("Time taken:", end-start)
    
    # Send the response back to the user
    emit('response', {'message': response})

def generate_chatbot_response(message):
    docs = rag.search_docs(by="embedding", query=message)
    
    # Use cummulative document scores to pass relevant documents to LLM chatbot
    cum_prob = 0.0
    combined_text = ""
    for i, doc in enumerate(docs):
        combined_text += f"\n--- Document ---\n{doc['text']}\n"
        cum_prob += doc['score']
        if cum_prob > 4.50:
            print("Relevant documents:", i+1)
            break
            
    res = rag.generate_query_output(query=message, context=combined_text)
    
    return res

if __name__ == '__main__':
    socketio.run(app, debug=True)
