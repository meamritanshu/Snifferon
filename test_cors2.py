from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins=["http://127.0.0.1:5001", "http://localhost:5001"])

@app.route('/')
def index():
    return render_template_string('''
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.2/socket.io.js"></script>
        <script>
            const socket = io();
            socket.on('connect', () => console.log("connected"));
        </script>
        hello
    ''')

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5001, allow_unsafe_werkzeug=True)
