from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Detectr+ is running!"

if __name__ == '__main__':
    print("Starting Detectr+ Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)