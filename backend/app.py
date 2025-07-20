from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import uuid
from datetime import datetime
import pdfplumber
from docx import Document
import spacy
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE = 'resume_auditor.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

# Load sentence transformer model
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    print("Sentence transformer model not available")
    sentence_model = None

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create resumes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            parsed_text TEXT NOT NULL,
            ats_score INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create job_descriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            text TEXT NOT NULL,
            FOREIGN KEY (resume_id) REFERENCES resumes (id)
        )
    ''')
    
    # Create improvements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS improvements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            section TEXT,
            suggestion TEXT,
            rating INTEGER,
            FOREIGN KEY (resume_id) REFERENCES resumes (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def analyze_ats_compliance(text):
    """Analyze resume for ATS compliance"""
    score = 0
    feedback = []
    
    # Check for contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    if re.search(email_pattern, text):
        score += 15
    else:
        feedback.append("Add a professional email address")
    
    if re.search(phone_pattern, text):
        score += 10
    else:
        feedback.append("Include a phone number")
    
    # Check for key sections
    sections = {
        'experience': ['experience', 'work history', 'employment', 'professional experience'],
        'education': ['education', 'academic', 'degree', 'university', 'college'],
        'skills': ['skills', 'technical skills', 'competencies', 'abilities']
    }
    
    text_lower = text.lower()
    
    for section, keywords in sections.items():
        if any(keyword in text_lower for keyword in keywords):
            score += 20
        else:
            feedback.append(f"Add a clear {section.title()} section")
    
    # Check for bullet points
    if '•' in text or '*' in text or re.search(r'^\s*[-\u2022\u2023\u25E6]', text, re.MULTILINE):
        score += 15
    else:
        feedback.append("Use bullet points to improve readability")
    
    # Check for problematic elements
    if 'table' in text_lower or 'image' in text_lower:
        feedback.append("Avoid tables and images for better ATS compatibility")
    else:
        score += 10
    
    # Length check
    word_count = len(text.split())
    if 300 <= word_count <= 800:
        score += 10
    else:
        feedback.append("Optimal resume length is 300-800 words")
    
    return min(score, 100), feedback

def extract_keywords_from_job_description(job_desc):
    """Extract keywords from job description using NLP"""
    if not nlp:
        return []
    
    doc = nlp(job_desc)
    keywords = []
    
    # Extract named entities and noun phrases
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'SKILL']:
            keywords.append(ent.text.lower())
    
    # Extract important nouns and adjectives
    for token in doc:
        if (token.pos_ in ['NOUN', 'ADJ'] and 
            len(token.text) > 2 and 
            not token.is_stop and 
            token.is_alpha):
            keywords.append(token.lemma_.lower())
    
    return list(set(keywords))

def calculate_keyword_match(resume_text, job_keywords):
    """Calculate keyword matching between resume and job description"""
    resume_lower = resume_text.lower()
    matched_keywords = []
    missing_keywords = []
    
    for keyword in job_keywords:
        if keyword in resume_lower:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    match_percentage = (len(matched_keywords) / len(job_keywords) * 100) if job_keywords else 0
    
    return {
        'matched': matched_keywords,
        'missing': missing_keywords,
        'match_percentage': round(match_percentage, 2)
    }

def generate_ai_suggestions(ats_score, keyword_analysis, feedback):
    """Generate AI-powered improvement suggestions"""
    suggestions = []
    
    # ATS-based suggestions
    if ats_score < 70:
        suggestions.append({
            'category': 'ATS Compliance',
            'priority': 'High',
            'suggestion': 'Your resume needs significant ATS optimization. Focus on adding missing sections and improving formatting.'
        })
    
    # Keyword-based suggestions
    if keyword_analysis['match_percentage'] < 50:
        suggestions.append({
            'category': 'Keyword Optimization',
            'priority': 'High',
            'suggestion': f"Add these missing keywords: {', '.join(keyword_analysis['missing'][:5])}"
        })
    
    # General feedback
    for item in feedback:
        suggestions.append({
            'category': 'General',
            'priority': 'Medium',
            'suggestion': item
        })
    
    return suggestions

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and parsing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique filename
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    if file_extension not in ['pdf', 'docx']:
        return jsonify({'error': 'Only PDF and DOCX files are supported'}), 400
    
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # Extract text based on file type
    if file_extension == 'pdf':
        parsed_text = extract_text_from_pdf(file_path)
    else:
        parsed_text = extract_text_from_docx(file_path)
    
    if not parsed_text.strip():
        return jsonify({'error': 'Could not extract text from file'}), 400
    
    # Analyze ATS compliance
    ats_score, feedback = analyze_ats_compliance(parsed_text)
    
    # Save to database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO resumes (file_name, parsed_text, ats_score)
        VALUES (?, ?, ?)
    ''', (file.filename, parsed_text, ats_score))
    resume_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Clean up uploaded file
    os.remove(file_path)
    
    return jsonify({
        'resume_id': resume_id,
        'ats_score': ats_score,
        'feedback': feedback,
        'text_preview': parsed_text[:500] + '...' if len(parsed_text) > 500 else parsed_text
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Analyze resume against job description"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    job_description = data.get('job_description', '')
    
    if not resume_id:
        return jsonify({'error': 'Resume ID required'}), 400
    
    # Get resume from database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT parsed_text, ats_score FROM resumes WHERE id = ?', (resume_id,))
    result = cursor.fetchone()
    
    if not result:
        return jsonify({'error': 'Resume not found'}), 404
    
    parsed_text, ats_score = result
    
    # Save job description
    if job_description:
        cursor.execute('''
            INSERT INTO job_descriptions (resume_id, text)
            VALUES (?, ?)
        ''', (resume_id, job_description))
        conn.commit()
    
    conn.close()
    
    # Analyze keywords if job description provided
    keyword_analysis = {'matched': [], 'missing': [], 'match_percentage': 0}
    if job_description:
        job_keywords = extract_keywords_from_job_description(job_description)
        keyword_analysis = calculate_keyword_match(parsed_text, job_keywords)
    
    # Generate suggestions
    ats_score_val, feedback = analyze_ats_compliance(parsed_text)
    suggestions = generate_ai_suggestions(ats_score_val, keyword_analysis, feedback)
    
    return jsonify({
        'resume_id': resume_id,
        'ats_score': ats_score_val,
        'keyword_analysis': keyword_analysis,
        'suggestions': suggestions,
        'overall_rating': 'Excellent' if ats_score_val >= 80 else 'Good' if ats_score_val >= 60 else 'Needs Improvement'
    })

@app.route('/api/resume/<int:resume_id>', methods=['GET'])
def get_resume(resume_id):
    """Get resume analysis results"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.file_name, r.parsed_text, r.ats_score, r.uploaded_at,
               jd.text as job_description
        FROM resumes r
        LEFT JOIN job_descriptions jd ON r.id = jd.resume_id
        WHERE r.id = ?
    ''', (resume_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'Resume not found'}), 404
    
    return jsonify({
        'file_name': result[0],
        'text_preview': result[1][:500] + '...' if len(result[1]) > 500 else result[1],
        'ats_score': result[2],
        'uploaded_at': result[3],
        'job_description': result[4]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)