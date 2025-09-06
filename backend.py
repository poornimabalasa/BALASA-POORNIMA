# backend.py
from flask import Flask, request, jsonify ,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
app = Flask '__name__'
CORS(app)
def preprocess_code(code):
    # Simple normalization: remove spaces and convert to lowercase
    return ''.join(code.split()).lower()
    @app.route('/')
    def home():
        #This will load templates/frontend.html
        return render_template('frontend.html')
@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    code1 = request.json.get('code1', '')
    code2 = request.json.get('code2', '')
    # Preprocess the codes
    code1_norm = preprocess_code(code1)
    code2_norm = preprocess_code(code2)
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform([code1_norm, code2_norm])
    vectors = vectorizer.toarray()
    # Calculate similarity
    sim_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return jsonify({'similarity_percentage': sim_score * 100})
if ('__name__') == '_main_':
    app.run(debug=True) 
