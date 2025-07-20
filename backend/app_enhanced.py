from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import uuid
import hashlib
from datetime import datetime
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import logging

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE = 'resume_auditor_enhanced.db'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.ats_model = None
        self.model_path = os.path.join(MODEL_FOLDER, 'ats_model.pkl')
        self.vectorizer_path = os.path.join(MODEL_FOLDER, 'vectorizer.pkl')
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.ats_model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Loaded existing ML model")
            else:
                self.ats_model = RandomForestRegressor(n_estimators=100, random_state=42)
                logger.info("Created new ML model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.ats_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.ats_model, f)
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def extract_features(self, text):
        """Extract features from resume text"""
        features = {}
        
        # Basic text features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Contact information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        features['has_email'] = 1 if re.search(email_pattern, text) else 0
        features['has_phone'] = 1 if re.search(phone_pattern, text) else 0
        
        # Section presence
        text_lower = text.lower()
        sections = {
            'experience': ['experience', 'work history', 'employment'],
            'education': ['education', 'academic', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies']
        }
        
        for section, keywords in sections.items():
            features[f'has_{section}'] = 1 if any(kw in text_lower for kw in keywords) else 0
        
        # Formatting features
        features['has_bullets'] = 1 if ('•' in text or '*' in text or 
                                       re.search(r'^\s*[-\u2022\u2023\u25E6]', text, re.MULTILINE)) else 0
        
        return features
    
    def train_model(self):
        """Train the ML model with historical data"""
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            
            # Get training data
            cursor.execute('''
                SELECT parsed_text, ats_score, user_feedback_score 
                FROM resumes 
                WHERE ats_score IS NOT NULL AND user_feedback_score IS NOT NULL
            ''')
            
            data = cursor.fetchall()
            conn.close()
            
            if len(data) < 10:  # Need minimum data for training
                logger.info("Insufficient data for training. Using rule-based scoring.")
                return False
            
            # Prepare features and targets
            texts = [row[0] for row in data]
            scores = [row[1] for row in data]
            feedback_scores = [row[2] for row in data]
            
            # Combine ATS score and user feedback
            target_scores = [(ats + feedback) / 2 for ats, feedback in zip(scores, feedback_scores)]
            
            # Extract features
            feature_list = []
            for text in texts:
                features = self.extract_features(text)
                feature_list.append(list(features.values()))
            
            # Vectorize text
            text_vectors = self.vectorizer.fit_transform(texts).toarray()
            
            # Combine features
            X = np.hstack([feature_list, text_vectors])
            y = np.array(target_scores)
            
            # Train model
            if len(X) > 0:
                self.ats_model.fit(X, y)
                self.save_model()
                logger.info(f"Model trained with {len(X)} samples")
                return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_ats_score(self, text):
        """Predict ATS score using ML model"""
        try:
            features = self.extract_features(text)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Check if model is trained
            if hasattr(self.ats_model, 'feature_importances_'):
                text_vector = self.vectorizer.transform([text]).toarray()
                X = np.hstack([feature_vector, text_vector])
                predicted_score = self.ats_model.predict(X)[0]
                return max(0, min(100, predicted_score))
            else:
                # Fallback to rule-based scoring
                return self.rule_based_scoring(text, features)
                
        except Exception as e:
            logger.error(f"Error predicting score: {e}")
            return self.rule_based_scoring(text, self.extract_features(text))
    
    def rule_based_scoring(self, text, features):
        """Fallback rule-based scoring"""
        score = 0
        
        # Contact information (25 points)
        score += features['has_email'] * 15
        score += features['has_phone'] * 10
        
        # Sections (45 points)
        score += features['has_experience'] * 20
        score += features['has_education'] * 15
        score += features['has_skills'] * 10
        
        # Formatting (20 points)
        score += features['has_bullets'] * 15
        
        # Length optimization (10 points)
        if 300 <= features['word_count'] <= 800:
            score += 10
        
        return min(score, 100)

# Initialize ML analyzer
ml_analyzer = MLResumeAnalyzer()

def init_db():
    """Initialize the enhanced database with ML tracking"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Enhanced resumes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            parsed_text TEXT NOT NULL,
            ats_score INTEGER,
            ml_predicted_score REAL,
            user_feedback_score INTEGER,
            keyword_match_percentage REAL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_analyzed TIMESTAMP,
            analysis_count INTEGER DEFAULT 1
        )
    ''')
    
    # Job descriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            text TEXT NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_id) REFERENCES resumes (id)
        )
    ''')
    
    # Enhanced improvements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS improvements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            category TEXT,
            suggestion TEXT,
            priority TEXT,
            implemented BOOLEAN DEFAULT FALSE,
            user_rating INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_id) REFERENCES resumes (id)
        )
    ''')
    
    # Resume edits tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resume_edits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            original_text TEXT,
            edited_text TEXT,
            edit_type TEXT,
            improvement_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resume_id) REFERENCES resumes (id)
        )
    ''')
    
    # ML training data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_training_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_samples INTEGER,
            model_accuracy REAL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def calculate_file_hash(file_content):
    """Calculate SHA-256 hash for file content integrity"""
    return hashlib.sha256(file_content.encode('utf-8')).hexdigest()

def extract_text_from_file(file_path, file_extension):
    """Extract text from uploaded files with error handling"""
    try:
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == 'pdf':
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text.strip()
            except ImportError:
                return "PDF text extraction requires pdfplumber library."
        elif file_extension == 'docx':
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            except ImportError:
                return "DOCX text extraction requires python-docx library."
        else:
            return "Unsupported file format"
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return f"Error extracting text from file: {str(e)}"

def enhanced_keyword_analysis(resume_text, job_description):
    """Enhanced keyword analysis with ML"""
    try:
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        
        # Fit on job description to get relevant keywords
        job_vector = vectorizer.fit_transform([job_description])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get keyword scores
        keyword_scores = job_vector.toarray()[0]
        keywords = [(feature_names[i], keyword_scores[i]) for i in range(len(feature_names))]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        
        # Check which keywords are in resume
        resume_lower = resume_text.lower()
        matched_keywords = []
        missing_keywords = []
        
        for keyword, score in keywords:
            if keyword in resume_lower:
                matched_keywords.append({'keyword': keyword, 'score': score})
            else:
                missing_keywords.append({'keyword': keyword, 'score': score})
        
        match_percentage = (len(matched_keywords) / len(keywords) * 100) if keywords else 0
        
        return {
            'matched': matched_keywords,
            'missing': missing_keywords,
            'match_percentage': round(match_percentage, 2),
            'total_keywords': len(keywords)
        }
    except Exception as e:
        logger.error(f"Error in keyword analysis: {e}")
        return {'matched': [], 'missing': [], 'match_percentage': 0, 'total_keywords': 0}

def generate_smart_suggestions(resume_text, ats_score, keyword_analysis):
    """Generate intelligent suggestions based on ML analysis"""
    suggestions = []
    
    # ATS Score based suggestions
    if ats_score < 60:
        suggestions.append({
            'category': 'Critical',
            'priority': 'High',
            'suggestion': 'Your resume needs major ATS optimization. Focus on structure and formatting.',
            'impact': 'High'
        })
    elif ats_score < 80:
        suggestions.append({
            'category': 'ATS Optimization',
            'priority': 'Medium',
            'suggestion': 'Good foundation, but some improvements needed for better ATS compatibility.',
            'impact': 'Medium'
        })
    
    # Keyword optimization
    if keyword_analysis['match_percentage'] < 40:
        top_missing = keyword_analysis['missing'][:3]
        keywords_text = ', '.join([kw['keyword'] for kw in top_missing])
        suggestions.append({
            'category': 'Keywords',
            'priority': 'High',
            'suggestion': f'Add these important keywords: {keywords_text}',
            'impact': 'High'
        })
    
    # Content analysis
    word_count = len(resume_text.split())
    if word_count < 300:
        suggestions.append({
            'category': 'Content',
            'priority': 'Medium',
            'suggestion': 'Your resume is too short. Add more details about your experience and achievements.',
            'impact': 'Medium'
        })
    elif word_count > 800:
        suggestions.append({
            'category': 'Content',
            'priority': 'Medium',
            'suggestion': 'Your resume is too long. Consider condensing to the most relevant information.',
            'impact': 'Medium'
        })
    
    return suggestions

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    """Enhanced resume upload with ML analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'txt'
    if file_extension not in ['pdf', 'docx', 'txt']:
        return jsonify({'error': 'Only PDF, DOCX, and TXT files are supported'}), 400
    
    try:
        # Save file
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Extract text
        parsed_text = extract_text_from_file(file_path, file_extension)
        
        if "Error" in parsed_text or "requires" in parsed_text:
            return jsonify({'error': parsed_text}), 400
        
        # Calculate file hash for integrity
        file_hash = calculate_file_hash(parsed_text)
        
        # Check for duplicate
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM resumes WHERE file_hash = ?', (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            os.remove(file_path)  # Remove duplicate file
            return jsonify({
                'message': 'This resume has been uploaded before',
                'resume_id': existing[0],
                'duplicate': True
            })
        
        # ML-powered ATS analysis
        ml_score = ml_analyzer.predict_ats_score(parsed_text)
        
        # Store in database
        cursor.execute('''
            INSERT INTO resumes (file_name, file_hash, parsed_text, ats_score, ml_predicted_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, file_hash, parsed_text, int(ml_score), ml_score))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Clean up file
        os.remove(file_path)
        
        return jsonify({
            'message': 'Resume uploaded and analyzed successfully',
            'resume_id': resume_id,
            'ats_score': int(ml_score),
            'ml_confidence': 'High' if hasattr(ml_analyzer.ats_model, 'feature_importances_') else 'Rule-based',
            'preview': parsed_text[:500] + "..." if len(parsed_text) > 500 else parsed_text
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Enhanced analysis with ML-powered insights"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    job_description = data.get('job_description', '')
    
    if not resume_id:
        return jsonify({'error': 'Resume ID required'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get resume
        cursor.execute('SELECT parsed_text, ats_score, ml_predicted_score FROM resumes WHERE id = ?', (resume_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'Resume not found'}), 404
        
        parsed_text, ats_score, ml_score = result
        
        # Enhanced keyword analysis
        keyword_analysis = enhanced_keyword_analysis(parsed_text, job_description) if job_description else {
            'matched': [], 'missing': [], 'match_percentage': 0, 'total_keywords': 0
        }
        
        # Generate smart suggestions
        suggestions = generate_smart_suggestions(parsed_text, ats_score, keyword_analysis)
        
        # Store job description and analysis
        if job_description:
            cursor.execute('''
                INSERT OR REPLACE INTO job_descriptions (resume_id, text, keywords)
                VALUES (?, ?, ?)
            ''', (resume_id, job_description, json.dumps(keyword_analysis)))
        
        # Store suggestions
        for suggestion in suggestions:
            cursor.execute('''
                INSERT INTO improvements (resume_id, category, suggestion, priority)
                VALUES (?, ?, ?, ?)
            ''', (resume_id, suggestion['category'], suggestion['suggestion'], suggestion['priority']))
        
        # Update analysis count
        cursor.execute('''
            UPDATE resumes 
            SET last_analyzed = CURRENT_TIMESTAMP, analysis_count = analysis_count + 1,
                keyword_match_percentage = ?
            WHERE id = ?
        ''', (keyword_analysis['match_percentage'], resume_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'resume_id': resume_id,
            'ats_score': ats_score,
            'ml_score': round(ml_score, 1),
            'keyword_analysis': keyword_analysis,
            'suggestions': suggestions,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/edit-resume', methods=['POST'])
def edit_resume():
    """AI-powered resume editing suggestions"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    edit_type = data.get('edit_type', 'general')  # general, keywords, formatting, content
    
    if not resume_id:
        return jsonify({'error': 'Resume ID required'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT parsed_text FROM resumes WHERE id = ?', (resume_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'Resume not found'}), 404
        
        original_text = result[0]
        
        # Generate edit suggestions based on type
        edit_suggestions = []
        
        if edit_type == 'keywords':
            # Get missing keywords from latest job description
            cursor.execute('''
                SELECT keywords FROM job_descriptions 
                WHERE resume_id = ? 
                ORDER BY created_at DESC LIMIT 1
            ''', (resume_id,))
            
            keywords_result = cursor.fetchone()
            if keywords_result:
                keywords_data = json.loads(keywords_result[0])
                missing_keywords = keywords_data.get('missing', [])[:5]
                
                for kw_data in missing_keywords:
                    keyword = kw_data['keyword']
                    edit_suggestions.append({
                        'type': 'keyword_addition',
                        'suggestion': f"Consider adding '{keyword}' to your skills or experience section",
                        'keyword': keyword,
                        'importance': kw_data['score']
                    })
        
        elif edit_type == 'formatting':
            # Analyze formatting issues
            if not re.search(r'^\s*[-\u2022\u2023\u25E6]', original_text, re.MULTILINE):
                edit_suggestions.append({
                    'type': 'formatting',
                    'suggestion': 'Add bullet points to improve readability',
                    'example': '• Managed team of 5 developers\n• Increased efficiency by 30%'
                })
            
            # Check for section headers
            if 'EXPERIENCE' not in original_text.upper():
                edit_suggestions.append({
                    'type': 'structure',
                    'suggestion': 'Add clear section headers like "PROFESSIONAL EXPERIENCE"',
                    'example': 'PROFESSIONAL EXPERIENCE\n========================'
                })
        
        elif edit_type == 'content':
            # Content improvement suggestions
            word_count = len(original_text.split())
            if word_count < 300:
                edit_suggestions.append({
                    'type': 'content_expansion',
                    'suggestion': 'Expand your experience descriptions with specific achievements and metrics',
                    'example': 'Instead of "Managed projects" try "Managed 3 cross-functional projects, delivering 20% ahead of schedule"'
                })
        
        conn.close()
        
        return jsonify({
            'resume_id': resume_id,
            'edit_type': edit_type,
            'suggestions': edit_suggestions,
            'original_length': len(original_text.split()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in edit suggestions: {e}")
        return jsonify({'error': f'Edit suggestion failed: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Collect user feedback for ML improvement"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    feedback_score = data.get('feedback_score')  # 1-5 rating
    improvement_ratings = data.get('improvement_ratings', {})  # Dict of improvement_id: rating
    
    if not resume_id or feedback_score is None:
        return jsonify({'error': 'Resume ID and feedback score required'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Update resume with user feedback
        cursor.execute('''
            UPDATE resumes 
            SET user_feedback_score = ?
            WHERE id = ?
        ''', (feedback_score * 20, resume_id))  # Convert 1-5 to 20-100 scale
        
        # Update improvement ratings
        for improvement_id, rating in improvement_ratings.items():
            cursor.execute('''
                UPDATE improvements 
                SET user_rating = ?
                WHERE id = ? AND resume_id = ?
            ''', (rating, improvement_id, resume_id))
        
        conn.commit()
        conn.close()
        
        # Retrain model with new feedback
        ml_analyzer.train_model()
        
        return jsonify({
            'message': 'Feedback submitted successfully',
            'model_updated': True
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({'error': f'Feedback submission failed: {str(e)}'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get application statistics and ML model performance"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM resumes')
        total_resumes = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(ats_score) FROM resumes WHERE ats_score IS NOT NULL')
        avg_ats_score = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(user_feedback_score) FROM resumes WHERE user_feedback_score IS NOT NULL')
        avg_user_score = cursor.fetchone()[0] or 0
        
        # ML model stats
        cursor.execute('SELECT COUNT(*) FROM resumes WHERE user_feedback_score IS NOT NULL')
        training_samples = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_resumes': total_resumes,
            'average_ats_score': round(avg_ats_score, 1),
            'average_user_score': round(avg_user_score, 1),
            'ml_training_samples': training_samples,
            'model_status': 'Trained' if hasattr(ml_analyzer.ats_model, 'feature_importances_') else 'Rule-based'
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': f'Stats retrieval failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': ml_analyzer.ats_model is not None,
        'database_connected': True,
        'timestamp': datetime.now().isoformat()
    })

# Initialize database and train model on startup
if __name__ == '__main__':
    init_db()
    ml_analyzer.train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)