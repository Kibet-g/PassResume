from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import uuid
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE = 'resume_auditor.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def extract_text_from_file(file_path, file_extension):
    """Simple text extraction for demo purposes"""
    try:
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # For demo purposes, return a sample text
            return """
            John Doe
            Software Engineer
            Email: john.doe@email.com
            Phone: (555) 123-4567
            
            EXPERIENCE
            Senior Software Engineer at Tech Corp (2020-2023)
            • Developed web applications using Python and JavaScript
            • Led a team of 5 developers
            • Improved system performance by 40%
            
            Software Engineer at StartupXYZ (2018-2020)
            • Built REST APIs using Flask and Django
            • Implemented automated testing procedures
            • Collaborated with cross-functional teams
            
            EDUCATION
            Bachelor of Science in Computer Science
            University of Technology (2014-2018)
            
            SKILLS
            Python, JavaScript, React, Flask, Django, SQL, Git
            """
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

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
    """Simple keyword extraction"""
    # Basic keyword extraction
    words = job_desc.lower().split()
    # Filter out common words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    keywords = [word.strip('.,!?;:') for word in words if len(word) > 3 and word not in stop_words]
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
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'txt'
    if file_extension not in ['pdf', 'docx', 'txt']:
        return jsonify({'error': 'Only PDF, DOCX, and TXT files are supported'}), 400
    
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # Extract text based on file type
    parsed_text = extract_text_from_file(file_path, file_extension)
    
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