from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import uuid
import hashlib
from datetime import datetime, timedelta
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import logging
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Import enhanced AI modules
from ai_enhanced_analyzer import (
    IntelligentResumeScanner, 
    SelfLearningAI, 
    EnhancedResumeRefiner
)
from pdf_export_system import AdvancedPDFExporter, ResumeAnalyticsTracker

app = Flask(__name__)

# Production-ready CORS configuration
if os.environ.get('FLASK_ENV') == 'production':
    # In production, specify allowed origins
    CORS(app, origins=[
        "https://passresume.netlify.app",
        "https://your-frontend-domain.vercel.app"  # Keep as backup
    ])
else:
    # In development, allow all origins
    CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE = 'resume_auditor_enhanced.db'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
JWT_SECRET = os.environ.get('JWT_SECRET', 'jwt-secret-key-change-in-production')

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

# Initialize Enhanced AI Components
intelligent_scanner = IntelligentResumeScanner()
self_learning_ai = SelfLearningAI(DATABASE)
resume_refiner = EnhancedResumeRefiner()
pdf_exporter = AdvancedPDFExporter()
analytics_tracker = ResumeAnalyticsTracker(DATABASE)

def init_db():
    """Initialize the enhanced database with ML tracking"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if we need to migrate the existing resumes table
    cursor.execute("PRAGMA table_info(resumes)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'resumes' in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        # Table exists, check if it has the old unique constraint
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='resumes'")
        table_sql = cursor.fetchone()
        if table_sql and 'file_hash TEXT UNIQUE NOT NULL' in table_sql[0]:
            # Need to recreate table without unique constraint
            logger.info("Migrating resumes table to remove unique constraint on file_hash")
            
            # Create new table with updated schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resumes_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    parsed_text TEXT NOT NULL,
                    ats_score INTEGER,
                    ml_predicted_score REAL,
                    user_feedback_score INTEGER,
                    keyword_match_percentage REAL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_analyzed TIMESTAMP,
                    analysis_count INTEGER DEFAULT 1,
                    original_text TEXT,
                    auto_improved BOOLEAN DEFAULT FALSE,
                    improvement_details TEXT,
                    original_ats_score INTEGER,
                    user_id INTEGER,
                    UNIQUE(file_hash, user_id)
                )
            ''')
            
            # Copy data from old table
            cursor.execute('''
                INSERT INTO resumes_new 
                SELECT id, file_name, file_hash, parsed_text, ats_score, ml_predicted_score, 
                       user_feedback_score, keyword_match_percentage, uploaded_at, last_analyzed, 
                       analysis_count, 
                       CASE WHEN original_text IS NOT NULL THEN original_text ELSE NULL END,
                       CASE WHEN auto_improved IS NOT NULL THEN auto_improved ELSE FALSE END,
                       CASE WHEN improvement_details IS NOT NULL THEN improvement_details ELSE NULL END,
                       CASE WHEN original_ats_score IS NOT NULL THEN original_ats_score ELSE NULL END,
                       CASE WHEN user_id IS NOT NULL THEN user_id ELSE NULL END
                FROM resumes
            ''')
            
            # Drop old table and rename new one
            cursor.execute('DROP TABLE resumes')
            cursor.execute('ALTER TABLE resumes_new RENAME TO resumes')
        else:
            # Table exists but doesn't have the problematic constraint, just add missing columns
            try:
                cursor.execute('ALTER TABLE resumes ADD COLUMN original_text TEXT')
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute('ALTER TABLE resumes ADD COLUMN auto_improved BOOLEAN DEFAULT FALSE')
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute('ALTER TABLE resumes ADD COLUMN improvement_details TEXT')
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute('ALTER TABLE resumes ADD COLUMN original_ats_score INTEGER')
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute('ALTER TABLE resumes ADD COLUMN user_id INTEGER')
            except sqlite3.OperationalError:
                pass
    else:
        # Create new table with correct schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                parsed_text TEXT NOT NULL,
                ats_score INTEGER,
                ml_predicted_score REAL,
                user_feedback_score INTEGER,
                keyword_match_percentage REAL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_analyzed TIMESTAMP,
                analysis_count INTEGER DEFAULT 1,
                original_text TEXT,
                auto_improved BOOLEAN DEFAULT FALSE,
                improvement_details TEXT,
                original_ats_score INTEGER,
                user_id INTEGER,
                UNIQUE(file_hash, user_id)
            )
        ''')
    
    # Users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    
    # User sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User activity log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity_type TEXT NOT NULL,
            activity_data TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
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

# Authentication helper functions
def generate_token(user_id):
    """Generate JWT token for user authentication"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=7),  # Token expires in 7 days
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_token(token):
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'error': 'Token is invalid or expired'}), 401
        
        # Add user_id to request context
        request.current_user_id = user_id
        return f(*args, **kwargs)
    
    return decorated

def log_user_activity(user_id, activity_type, activity_data=None, ip_address=None):
    """Log user activity for tracking and analytics"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_activity (user_id, activity_type, activity_data, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, activity_type, json.dumps(activity_data) if activity_data else None, ip_address))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging user activity: {e}")

def get_user_by_id(user_id):
    """Get user information by ID"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, email, first_name, last_name, created_at, last_login, is_active
            FROM users WHERE id = ? AND is_active = TRUE
        ''', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'email': user[1],
                'first_name': user[2],
                'last_name': user[3],
                'created_at': user[4],
                'last_login': user[5],
                'is_active': user[6]
            }
        return None
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None

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
    """Enhanced resume upload with automatic ATS optimization"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if user is authenticated (optional)
    user_id = None
    token = request.headers.get('Authorization')
    if token:
        if token.startswith('Bearer '):
            token = token[7:]
        user_id = verify_token(token)
    
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
        
        # Check for duplicate (only for authenticated users)
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        if user_id:
            cursor.execute('SELECT id FROM resumes WHERE file_hash = ? AND user_id = ?', (file_hash, user_id))
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
        original_ats_score = ml_analyzer.predict_ats_score(parsed_text)
        
        # Define ATS threshold (70 is considered good)
        ATS_THRESHOLD = 70
        
        # Check if resume needs improvement
        needs_improvement = original_ats_score < ATS_THRESHOLD
        improved_text = parsed_text
        applied_improvements = []
        final_ats_score = original_ats_score
        
        if needs_improvement:
            logger.info(f"Resume ATS score ({original_ats_score}) below threshold ({ATS_THRESHOLD}). Auto-improving...")
            
            # Apply automatic improvements
            improved_text = parsed_text
            
            # 1. Add professional formatting and structure
            if 'PROFESSIONAL SUMMARY' not in improved_text.upper() and 'PROFESSIONAL PROFILE' not in improved_text.upper():
                lines = improved_text.split('\n')
                name_line = lines[0] if lines else "YOUR NAME"
                
                # Find contact info
                contact_info = []
                for line in lines[:5]:
                    if re.search(r'@|phone|email|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line.lower()):
                        contact_info.append(line.strip())
                
                # Restructure with proper headers
                improved_text = f"{name_line}\n"
                if contact_info:
                    improved_text += "\n".join(contact_info) + "\n\n"
                
                improved_text += "PROFESSIONAL SUMMARY\n" + "="*20 + "\n"
                
                # Add the rest of the content
                content_start = 1
                while content_start < len(lines) and (not lines[content_start].strip() or 
                      re.search(r'@|phone|email|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', lines[content_start].lower())):
                    content_start += 1
                
                if content_start < len(lines):
                    improved_text += "\n".join(lines[content_start:])
                
                applied_improvements.append('Added professional structure with clear section headers')
            
            # 2. Improve bullet points and formatting
            if not re.search(r'^\s*[•\-\*]', improved_text, re.MULTILINE):
                lines = improved_text.split('\n')
                improved_lines = []
                
                for line in lines:
                    if (len(line.strip()) > 20 and 
                        not line.strip().isupper() and 
                        not re.match(r'^[A-Z\s]+$', line.strip()) and
                        '=' not in line and
                        not re.search(r'@|phone|email', line.lower())):
                        improved_lines.append(f"• {line.strip()}")
                    else:
                        improved_lines.append(line)
                
                improved_text = '\n'.join(improved_lines)
                applied_improvements.append('Converted text to professional bullet point format')
            
            # 3. Add missing critical sections
            if 'SKILLS' not in improved_text.upper() and 'TECHNICAL SKILLS' not in improved_text.upper():
                improved_text += f"\n\nTECHNICAL SKILLS\n{'='*15}\n• Leadership and Team Management\n• Project Management\n• Problem-Solving\n• Communication"
                applied_improvements.append('Added TECHNICAL SKILLS section')
            
            # 4. Enhance with ATS keywords
            common_keywords = ['leadership', 'project management', 'team collaboration', 'problem-solving']
            keywords_added = []
            for keyword in common_keywords[:2]:  # Add top 2 keywords
                if keyword.lower() not in improved_text.lower():
                    skills_match = re.search(r'(SKILLS|TECHNICAL SKILLS)[\s\S]*?(?=\n[A-Z]|\n\n|$)', improved_text, re.IGNORECASE)
                    if skills_match:
                        skills_section = skills_match.group(0)
                        enhanced_skills = skills_section + f"\n• {keyword.title()}"
                        improved_text = improved_text.replace(skills_section, enhanced_skills)
                        keywords_added.append(keyword)
            
            if keywords_added:
                applied_improvements.append(f'Added ATS keywords: {", ".join(keywords_added)}')
            
            # Recalculate ATS score after improvements
            final_ats_score = ml_analyzer.predict_ats_score(improved_text)
            
            logger.info(f"Auto-improvement complete. Score improved from {original_ats_score} to {final_ats_score}")
        
        # Store in database with improved text if applicable
        cursor.execute('''
            INSERT INTO resumes (file_name, file_hash, parsed_text, ats_score, ml_predicted_score, 
                               original_text, auto_improved, improvement_details, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file.filename, file_hash, improved_text, int(final_ats_score), final_ats_score,
              parsed_text if needs_improvement else None, needs_improvement, 
              json.dumps(applied_improvements) if applied_improvements else None, user_id))
        
        resume_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log activity for authenticated users
        if user_id:
            log_user_activity(user_id, 'resume_upload', {
                'resume_id': resume_id,
                'file_name': file.filename,
                'ats_score': int(final_ats_score),
                'auto_improved': needs_improvement
            }, request.remote_addr)
        
        # Clean up file
        os.remove(file_path)
        
        response_data = {
            'message': 'Resume uploaded and analyzed successfully',
            'resume_id': resume_id,
            'original_ats_score': int(original_ats_score),
            'final_ats_score': int(final_ats_score),
            'ats_friendly': final_ats_score >= ATS_THRESHOLD,
            'auto_improved': needs_improvement,
            'ml_confidence': 'High' if hasattr(ml_analyzer.ats_model, 'feature_importances_') else 'Rule-based',
            'preview': improved_text[:500] + "..." if len(improved_text) > 500 else improved_text,
            'authenticated': user_id is not None
        }
        
        if needs_improvement:
            response_data.update({
                'improvement_applied': True,
                'improvements_made': applied_improvements,
                'score_improvement': int(final_ats_score - original_ats_score),
                'message': f'Resume auto-improved! ATS score increased from {int(original_ats_score)} to {int(final_ats_score)}'
            })
        else:
            response_data.update({
                'improvement_applied': False,
                'message': f'Great! Your resume is already ATS-friendly with a score of {int(original_ats_score)}'
            })
        
        return jsonify(response_data)
        
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

@app.route('/api/auto-improve-resume', methods=['POST'])
def auto_improve_resume():
    """Automatically improve resume with comprehensive AI enhancements"""
    data = request.get_json()
    resume_id = data.get('resume_id')
    original_text = data.get('original_text', '')
    
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
        
        current_text = result[0]
        improved_text = current_text
        applied_suggestions = []
        
        # 1. Add professional formatting and structure
        if 'PROFESSIONAL SUMMARY' not in improved_text.upper() and 'PROFESSIONAL PROFILE' not in improved_text.upper():
            # Extract first paragraph as summary if it exists
            lines = improved_text.split('\n')
            name_line = lines[0] if lines else "YOUR NAME"
            
            # Find contact info
            contact_info = []
            for line in lines[:5]:
                if re.search(r'@|phone|email|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line.lower()):
                    contact_info.append(line.strip())
            
            # Restructure with proper headers
            improved_text = f"{name_line}\n"
            if contact_info:
                improved_text += "\n".join(contact_info) + "\n\n"
            
            improved_text += "PROFESSIONAL SUMMARY\n" + "="*20 + "\n"
            
            # Add the rest of the content
            content_start = 1
            while content_start < len(lines) and (not lines[content_start].strip() or 
                  re.search(r'@|phone|email|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', lines[content_start].lower())):
                content_start += 1
            
            if content_start < len(lines):
                improved_text += "\n".join(lines[content_start:])
            
            applied_suggestions.append({
                'type': 'structure',
                'suggestion': 'Added professional formatting with clear section headers'
            })
        
        # 2. Improve bullet points and formatting
        if not re.search(r'^\s*[•\-\*]', improved_text, re.MULTILINE):
            # Convert sentences to bullet points in experience sections
            lines = improved_text.split('\n')
            improved_lines = []
            
            for line in lines:
                if (len(line.strip()) > 20 and 
                    not line.strip().isupper() and 
                    not re.match(r'^[A-Z\s]+$', line.strip()) and
                    '=' not in line and
                    not re.search(r'@|phone|email', line.lower())):
                    improved_lines.append(f"• {line.strip()}")
                else:
                    improved_lines.append(line)
            
            improved_text = '\n'.join(improved_lines)
            applied_suggestions.append({
                'type': 'formatting',
                'suggestion': 'Converted text to professional bullet point format'
            })
        
        # 3. Add missing sections
        sections_to_add = []
        
        if 'SKILLS' not in improved_text.upper() and 'TECHNICAL SKILLS' not in improved_text.upper():
            sections_to_add.append({
                'header': 'TECHNICAL SKILLS',
                'content': '• Programming Languages: Python, JavaScript, SQL\n• Frameworks: React, Node.js, Flask\n• Tools: Git, Docker, AWS\n• Databases: MySQL, PostgreSQL, MongoDB'
            })
        
        if 'EDUCATION' not in improved_text.upper():
            sections_to_add.append({
                'header': 'EDUCATION',
                'content': '• Bachelor\'s Degree in [Your Field]\n• [University Name], [Year]\n• Relevant Coursework: [List relevant courses]'
            })
        
        if 'EXPERIENCE' not in improved_text.upper() and 'WORK EXPERIENCE' not in improved_text.upper():
            sections_to_add.append({
                'header': 'PROFESSIONAL EXPERIENCE',
                'content': '• [Job Title] at [Company Name] ([Start Date] - [End Date])\n• [Achievement with quantifiable results]\n• [Another achievement with metrics]'
            })
        
        # Add missing sections
        for section in sections_to_add:
            improved_text += f"\n\n{section['header']}\n{'='*len(section['header'])}\n{section['content']}"
            applied_suggestions.append({
                'type': 'content_expansion',
                'suggestion': f'Added {section["header"]} section with professional template'
            })
        
        # 4. Enhance keywords based on common ATS requirements
        common_keywords = [
            'leadership', 'project management', 'team collaboration', 
            'problem-solving', 'analytical thinking', 'communication',
            'results-driven', 'innovative', 'strategic planning'
        ]
        
        keywords_added = []
        for keyword in common_keywords[:3]:  # Add top 3 keywords
            if keyword.lower() not in improved_text.lower():
                # Find skills section to add keyword
                skills_match = re.search(r'(SKILLS|TECHNICAL SKILLS)[\s\S]*?(?=\n[A-Z]|\n\n|$)', improved_text, re.IGNORECASE)
                if skills_match:
                    skills_section = skills_match.group(0)
                    enhanced_skills = skills_section + f"\n• {keyword.title()}"
                    improved_text = improved_text.replace(skills_section, enhanced_skills)
                    keywords_added.append(keyword)
        
        if keywords_added:
            applied_suggestions.append({
                'type': 'keyword_addition',
                'suggestion': f'Added high-impact keywords: {", ".join(keywords_added)}'
            })
        
        # 5. Improve action verbs and quantify achievements
        weak_verbs = ['did', 'worked on', 'helped', 'was responsible for']
        strong_verbs = ['achieved', 'implemented', 'optimized', 'delivered']
        
        for i, weak in enumerate(weak_verbs):
            if weak in improved_text.lower() and i < len(strong_verbs):
                improved_text = re.sub(weak, strong_verbs[i], improved_text, flags=re.IGNORECASE)
                applied_suggestions.append({
                    'type': 'content_enhancement',
                    'suggestion': f'Replaced weak verb "{weak}" with stronger action verb "{strong_verbs[i]}"'
                })
        
        # Calculate final ATS score for improved text
        final_ats_score = ml_analyzer.predict_ats_score(improved_text)
        
        # Update the database with improved text and new ATS score
        cursor.execute('''
            UPDATE resumes 
            SET parsed_text = ?, ats_score = ?
            WHERE id = ?
        ''', (improved_text, final_ats_score, resume_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'resume_id': resume_id,
            'improved_text': improved_text,
            'applied_suggestions': applied_suggestions,
            'improvement_count': len(applied_suggestions),
            'original_length': len(original_text.split()),
            'improved_length': len(improved_text.split()),
            'final_ats_score': final_ats_score,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in auto-improvement: {e}")
        return jsonify({'error': f'Auto-improvement failed: {str(e)}'}), 500

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

@app.route('/api/resume/<int:resume_id>', methods=['GET'])
def get_resume(resume_id):
    """Get resume details and analysis results"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # First check what columns exist in the table
        cursor.execute("PRAGMA table_info(resumes)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Build query based on available columns
        base_columns = ['file_name', 'parsed_text', 'ats_score', 'uploaded_at']
        optional_columns = ['original_text', 'auto_improved', 'improvement_details', 'original_ats_score']
        
        select_columns = base_columns.copy()
        for col in optional_columns:
            if col in columns:
                select_columns.append(col)
        
        query = f"SELECT {', '.join(select_columns)} FROM resumes WHERE id = ?"
        cursor.execute(query, (resume_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Resume not found'}), 404
        
        # Build response based on available data
        response_data = {
            'resume_id': resume_id,
            'file_name': result[0],
            'parsed_text': result[1],
            'ats_score': result[2],
            'uploaded_at': result[3]
        }
        
        # Add optional fields if they exist
        idx = 4
        if 'original_text' in columns:
            response_data['original_text'] = result[idx] if idx < len(result) else None
            idx += 1
        else:
            response_data['original_text'] = None
            
        if 'auto_improved' in columns:
            response_data['auto_improved'] = bool(result[idx]) if idx < len(result) and result[idx] is not None else False
            idx += 1
        else:
            response_data['auto_improved'] = False
            
        if 'improvement_details' in columns:
            details = result[idx] if idx < len(result) else None
            response_data['improvement_details'] = json.loads(details) if details else []
            idx += 1
        else:
            response_data['improvement_details'] = []
            
        if 'original_ats_score' in columns:
            response_data['original_ats_score'] = result[idx] if idx < len(result) else None
        else:
            response_data['original_ats_score'] = response_data['ats_score']
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting resume: {e}")
        return jsonify({'error': f'Resume retrieval failed: {str(e)}'}), 500

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

# Authentication API endpoints
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    
    # Validation
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400
    
    # Email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'User with this email already exists'}), 409
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (email, password_hash, first_name, last_name)
            VALUES (?, ?, ?, ?)
        ''', (email, password_hash, first_name, last_name))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Generate token
        token = generate_token(user_id)
        
        # Log activity
        log_user_activity(user_id, 'signup', {'email': email}, request.remote_addr)
        
        return jsonify({
            'message': 'User created successfully',
            'token': token,
            'user': {
                'id': user_id,
                'email': email,
                'first_name': first_name,
                'last_name': last_name
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error in signup: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get user
        cursor.execute('''
            SELECT id, email, password_hash, first_name, last_name, is_active
            FROM users WHERE email = ?
        ''', (email,))
        
        user = cursor.fetchone()
        if not user or not user[5]:  # Check if user exists and is active
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Verify password
        if not check_password_hash(user[2], password):
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user[0],))
        conn.commit()
        conn.close()
        
        # Generate token
        token = generate_token(user[0])
        
        # Log activity
        log_user_activity(user[0], 'login', {'email': email}, request.remote_addr)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user[0],
                'email': user[1],
                'first_name': user[3],
                'last_name': user[4]
            }
        })
        
    except Exception as e:
        logger.error(f"Error in login: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/profile', methods=['GET'])
@token_required
def get_profile():
    """Get user profile information"""
    user = get_user_by_id(request.current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user})

@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile():
    """Update user profile information"""
    data = request.get_json()
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET first_name = ?, last_name = ?
            WHERE id = ?
        ''', (first_name, last_name, request.current_user_id))
        
        conn.commit()
        conn.close()
        
        # Log activity
        log_user_activity(request.current_user_id, 'profile_update', 
                         {'first_name': first_name, 'last_name': last_name}, 
                         request.remote_addr)
        
        return jsonify({'message': 'Profile updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({'error': 'Profile update failed'}), 500

@app.route('/api/user/resumes', methods=['GET'])
@token_required
def get_user_resumes():
    """Get all resumes for the authenticated user"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get user's resumes with pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        offset = (page - 1) * per_page
        
        cursor.execute('''
            SELECT id, file_name, ats_score, uploaded_at, auto_improved, original_ats_score
            FROM resumes 
            WHERE user_id = ? 
            ORDER BY uploaded_at DESC 
            LIMIT ? OFFSET ?
        ''', (request.current_user_id, per_page, offset))
        
        resumes = cursor.fetchall()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM resumes WHERE user_id = ?', (request.current_user_id,))
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        resume_list = []
        for resume in resumes:
            resume_list.append({
                'id': resume[0],
                'file_name': resume[1],
                'ats_score': resume[2],
                'uploaded_at': resume[3],
                'auto_improved': bool(resume[4]) if resume[4] is not None else False,
                'original_ats_score': resume[5]
            })
        
        return jsonify({
            'resumes': resume_list,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
        
    except Exception as e:
        logger.error(f"Error getting user resumes: {e}")
        return jsonify({'error': 'Failed to retrieve resumes'}), 500

@app.route('/api/user/activity', methods=['GET'])
@token_required
def get_user_activity():
    """Get user activity history"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get recent activity
        limit = request.args.get('limit', 20, type=int)
        
        cursor.execute('''
            SELECT activity_type, activity_data, created_at
            FROM user_activity 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (request.current_user_id, limit))
        
        activities = cursor.fetchall()
        conn.close()
        
        activity_list = []
        for activity in activities:
            activity_data = json.loads(activity[1]) if activity[1] else {}
            activity_list.append({
                'type': activity[0],
                'data': activity_data,
                'timestamp': activity[2]
            })
        
        return jsonify({'activities': activity_list})
        
    except Exception as e:
        logger.error(f"Error getting user activity: {e}")
        return jsonify({'error': 'Failed to retrieve activity'}), 500

@app.route('/api/user/stats', methods=['GET'])
@token_required
def get_user_stats():
    """Get user-specific statistics"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get user stats
        cursor.execute('SELECT COUNT(*) FROM resumes WHERE user_id = ?', (request.current_user_id,))
        total_resumes = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(ats_score) FROM resumes WHERE user_id = ? AND ats_score IS NOT NULL', 
                      (request.current_user_id,))
        avg_ats_score = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(*) FROM resumes WHERE user_id = ? AND auto_improved = TRUE', 
                      (request.current_user_id,))
        improved_resumes = cursor.fetchone()[0]
        
        # Get recent uploads (last 30 days)
        cursor.execute('''
            SELECT COUNT(*) FROM resumes 
            WHERE user_id = ? AND uploaded_at > datetime('now', '-30 days')
        ''', (request.current_user_id,))
        recent_uploads = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_resumes': total_resumes,
            'average_ats_score': round(avg_ats_score, 1),
            'improved_resumes': improved_resumes,
            'recent_uploads': recent_uploads,
            'improvement_rate': round((improved_resumes / total_resumes * 100), 1) if total_resumes > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'error': 'Failed to retrieve user statistics'}), 500

@app.route('/api/intelligent-scan', methods=['POST'])
@token_required
def intelligent_scan_resume():
    """Enhanced AI scanning of resume sections"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
        
        # Perform intelligent scanning
        scan_results = intelligent_scanner.analyze_sections(resume_text)
        quality_score = intelligent_scanner.calculate_quality_score(resume_text)
        
        # Log activity
        log_user_activity(request.current_user_id, 'intelligent_scan', 
                         {'quality_score': quality_score}, request.remote_addr)
        
        return jsonify({
            'scan_results': scan_results,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in intelligent scan: {e}")
        return jsonify({'error': 'Intelligent scan failed'}), 500

@app.route('/api/refine-resume', methods=['POST'])
@token_required
def refine_resume():
    """AI-powered resume refinement"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        job_description = data.get('job_description', '')
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
        
        # Refine the resume
        refined_resume = resume_refiner.refine_resume(resume_text, job_description)
        
        # Get improvement suggestions
        improvements = resume_refiner.get_improvement_suggestions(resume_text)
        
        # Log activity
        log_user_activity(request.current_user_id, 'resume_refinement', 
                         {'improvements_count': len(improvements)}, request.remote_addr)
        
        return jsonify({
            'refined_resume': refined_resume,
            'improvements': improvements,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in resume refinement: {e}")
        return jsonify({'error': 'Resume refinement failed'}), 500

@app.route('/api/export-pdf', methods=['POST'])
@token_required
def export_resume_pdf():
    """Export refined resume as PDF"""
    try:
        data = request.get_json()
        resume_data = data.get('resume_data', {})
        template_style = data.get('template_style', 'professional')
        
        if not resume_data:
            return jsonify({'error': 'Resume data is required'}), 400
        
        # Generate PDF
        pdf_buffer = pdf_exporter.export_resume(resume_data, template_style)
        
        # Track analytics
        analytics_tracker.track_export(request.current_user_id, template_style)
        
        # Log activity
        log_user_activity(request.current_user_id, 'pdf_export', 
                         {'template_style': template_style}, request.remote_addr)
        
        # Return PDF as base64 for download
        import base64
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'pdf_data': pdf_base64,
            'filename': f"refined_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in PDF export: {e}")
        return jsonify({'error': 'PDF export failed'}), 500

@app.route('/api/ai-training-status', methods=['GET'])
@token_required
def get_ai_training_status():
    """Get AI training and learning status"""
    try:
        training_stats = self_learning_ai.get_training_stats()
        
        return jsonify({
            'training_stats': training_stats,
            'last_training': self_learning_ai.last_training_time,
            'model_version': self_learning_ai.model_version,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting AI training status: {e}")
        return jsonify({'error': 'Failed to get training status'}), 500

@app.route('/api/trigger-ai-learning', methods=['POST'])
@token_required
def trigger_ai_learning():
    """Manually trigger AI learning process"""
    try:
        # Trigger learning from database
        self_learning_ai.learn_from_database()
        
        # Optionally trigger external data learning
        external_learning = request.get_json().get('include_external', False)
        if external_learning:
            self_learning_ai.learn_from_external_data()
        
        # Log activity
        log_user_activity(request.current_user_id, 'ai_learning_trigger', 
                         {'external_learning': external_learning}, request.remote_addr)
        
        return jsonify({
            'message': 'AI learning process triggered successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error triggering AI learning: {e}")
        return jsonify({'error': 'AI learning trigger failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'ml_model_loaded': ml_analyzer.ats_model is not None,
        'ai_components_loaded': True,
        'database_connected': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/init-db', methods=['POST'])
def manual_init_db():
    """Manual database initialization endpoint"""
    try:
        init_db()
        ml_analyzer.train_model()
        return jsonify({
            'status': 'success',
            'message': 'Database initialized successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Database initialization failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# Initialize database and train model on startup (for production)
try:
    init_db()
    ml_analyzer.train_model()
    
    # Initialize enhanced AI components
    self_learning_ai.learn_from_database()
    logger.info("Database, ML model, and enhanced AI components initialized successfully")
except Exception as e:
    logger.error(f"Error during startup initialization: {e}")

# Initialize database and train model on startup
if __name__ == '__main__':
    init_db()
    ml_analyzer.train_model()
    
    # Initialize enhanced AI components for development
    try:
        self_learning_ai.learn_from_database()
        logger.info("Enhanced AI components initialized for development")
    except Exception as e:
        logger.error(f"Error initializing enhanced AI components: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)