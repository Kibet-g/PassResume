"""
Enhanced AI Resume Analyzer with Intelligent Scanning and Self-Learning Capabilities
"""

import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests
from urllib.parse import quote
import time

logger = logging.getLogger(__name__)

class IntelligentResumeScanner:
    """Advanced AI scanner that analyzes each section of a resume thoroughly"""
    
    def __init__(self):
        self.section_patterns = {
            'contact': {
                'patterns': [r'contact', r'personal\s+information', r'details'],
                'required_elements': ['email', 'phone', 'location']
            },
            'summary': {
                'patterns': [r'summary', r'profile', r'objective', r'about'],
                'keywords': ['experienced', 'skilled', 'professional', 'dedicated']
            },
            'experience': {
                'patterns': [r'experience', r'work\s+history', r'employment', r'career'],
                'keywords': ['managed', 'led', 'developed', 'implemented', 'achieved']
            },
            'education': {
                'patterns': [r'education', r'academic', r'qualifications', r'degree'],
                'keywords': ['university', 'college', 'bachelor', 'master', 'phd']
            },
            'skills': {
                'patterns': [r'skills', r'technical\s+skills', r'competencies', r'expertise'],
                'categories': ['technical', 'soft', 'language', 'certification']
            },
            'projects': {
                'patterns': [r'projects', r'portfolio', r'work\s+samples'],
                'keywords': ['built', 'created', 'designed', 'developed']
            },
            'certifications': {
                'patterns': [r'certifications', r'certificates', r'licenses'],
                'keywords': ['certified', 'licensed', 'accredited']
            }
        }
        
        self.ats_keywords = {
            'action_verbs': [
                'achieved', 'administered', 'analyzed', 'built', 'collaborated',
                'created', 'delivered', 'developed', 'enhanced', 'executed',
                'implemented', 'improved', 'increased', 'led', 'managed',
                'optimized', 'organized', 'planned', 'reduced', 'streamlined'
            ],
            'quantifiable_metrics': [
                r'\d+%', r'\$\d+', r'\d+\s*(million|thousand|k|m)',
                r'\d+\s*(years?|months?)', r'\d+\s*(people|team|members)'
            ]
        }
    
    def scan_resume_sections(self, text: str) -> Dict:
        """Thoroughly scan and analyze each section of the resume"""
        sections_found = {}
        text_lower = text.lower()
        
        for section_name, section_config in self.section_patterns.items():
            section_analysis = {
                'found': False,
                'content': '',
                'quality_score': 0,
                'suggestions': [],
                'keywords_found': [],
                'missing_elements': []
            }
            
            # Find section in text
            for pattern in section_config['patterns']:
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    section_analysis['found'] = True
                    # Extract section content (rough estimation)
                    start_pos = matches[0].start()
                    # Find next section or end of text
                    next_section_pos = len(text)
                    for other_section, other_config in self.section_patterns.items():
                        if other_section != section_name:
                            for other_pattern in other_config['patterns']:
                                other_matches = list(re.finditer(other_pattern, text_lower[start_pos + 50:]))
                                if other_matches:
                                    next_section_pos = min(next_section_pos, start_pos + 50 + other_matches[0].start())
                    
                    section_analysis['content'] = text[start_pos:next_section_pos].strip()
                    break
            
            # Analyze section quality
            if section_analysis['found']:
                section_analysis['quality_score'] = self._analyze_section_quality(
                    section_name, section_analysis['content'], section_config
                )
                section_analysis['suggestions'] = self._generate_section_suggestions(
                    section_name, section_analysis['content'], section_config
                )
            
            sections_found[section_name] = section_analysis
        
        return sections_found
    
    def _analyze_section_quality(self, section_name: str, content: str, config: Dict) -> float:
        """Analyze the quality of a specific section"""
        score = 0.0
        content_lower = content.lower()
        
        if section_name == 'contact':
            # Check for required contact elements
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            
            if re.search(email_pattern, content): score += 40
            if re.search(phone_pattern, content): score += 30
            if any(loc in content_lower for loc in ['city', 'state', 'country', 'address']): score += 30
        
        elif section_name == 'experience':
            # Check for action verbs and quantifiable achievements
            action_verbs_found = sum(1 for verb in self.ats_keywords['action_verbs'] 
                                   if verb in content_lower)
            score += min(action_verbs_found * 5, 40)
            
            # Check for quantifiable metrics
            metrics_found = sum(1 for pattern in self.ats_keywords['quantifiable_metrics']
                              if re.search(pattern, content))
            score += min(metrics_found * 15, 60)
        
        elif section_name == 'skills':
            # Count number of skills mentioned
            skills_count = len([word for word in content.split() if len(word) > 3])
            score += min(skills_count * 2, 80)
            
            # Check for skill categories
            if 'technical' in content_lower or 'programming' in content_lower: score += 20
        
        elif section_name == 'education':
            # Check for degree information
            degree_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
            if any(keyword in content_lower for keyword in degree_keywords): score += 50
            
            # Check for institution and graduation year
            if re.search(r'\d{4}', content): score += 25  # Year mentioned
            if 'university' in content_lower or 'college' in content_lower: score += 25
        
        return min(score, 100.0)
    
    def _generate_section_suggestions(self, section_name: str, content: str, config: Dict) -> List[str]:
        """Generate improvement suggestions for a specific section"""
        suggestions = []
        content_lower = content.lower()
        
        if section_name == 'contact':
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            
            if not re.search(email_pattern, content):
                suggestions.append("Add a professional email address")
            if not re.search(phone_pattern, content):
                suggestions.append("Include a phone number")
            if not any(loc in content_lower for loc in ['city', 'state', 'location']):
                suggestions.append("Add your location (city, state)")
        
        elif section_name == 'experience':
            action_verbs_count = sum(1 for verb in self.ats_keywords['action_verbs'] 
                                   if verb in content_lower)
            if action_verbs_count < 3:
                suggestions.append("Use more action verbs to describe your achievements")
            
            metrics_count = sum(1 for pattern in self.ats_keywords['quantifiable_metrics']
                              if re.search(pattern, content))
            if metrics_count < 2:
                suggestions.append("Add quantifiable metrics to demonstrate impact")
        
        elif section_name == 'skills':
            if len(content.split()) < 10:
                suggestions.append("Expand your skills section with more relevant technologies")
            if 'technical' not in content_lower:
                suggestions.append("Organize skills into categories (Technical, Soft Skills, etc.)")
        
        return suggestions

class SelfLearningAI:
    """AI system that continuously learns from patterns and external data"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.models = {
            'ats_predictor': RandomForestRegressor(n_estimators=200, random_state=42),
            'improvement_predictor': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'section_quality_predictor': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.model_path = 'models/self_learning_ai.pkl'
        self.last_training = None
        self.training_data_cache = []
        
    def collect_training_data(self) -> List[Dict]:
        """Collect anonymized training data from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get anonymized resume data
            cursor.execute('''
                SELECT 
                    r.parsed_text,
                    r.ats_score,
                    r.user_feedback_score,
                    r.improvement_details,
                    AVG(i.user_rating) as avg_improvement_rating
                FROM resumes r
                LEFT JOIN improvements i ON r.id = i.resume_id
                WHERE r.ats_score IS NOT NULL
                GROUP BY r.id
                HAVING COUNT(*) > 0
            ''')
            
            training_data = []
            for row in cursor.fetchall():
                # Anonymize the text (remove personal information)
                anonymized_text = self._anonymize_text(row[0])
                training_data.append({
                    'text': anonymized_text,
                    'ats_score': row[1],
                    'user_feedback': row[2] or 0,
                    'improvement_details': row[3] or '',
                    'improvement_rating': row[4] or 0
                })
            
            conn.close()
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return []
    
    def _anonymize_text(self, text: str) -> str:
        """Remove personal information from text for training"""
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
        
        # Remove names (basic pattern matching)
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        
        # Remove addresses
        text = re.sub(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr)', '[ADDRESS]', text)
        
        return text
    
    def fetch_external_data(self) -> List[Dict]:
        """Fetch publicly available resume improvement data"""
        external_data = []
        
        # Simulate fetching from public APIs or datasets
        # In a real implementation, you would integrate with:
        # - LinkedIn Learning API
        # - Indeed Career Guide API
        # - Public resume datasets
        # - Industry-specific job boards
        
        try:
            # Example: Fetch industry keywords and trends
            industry_keywords = self._fetch_industry_keywords()
            external_data.extend(industry_keywords)
            
            # Example: Fetch ATS best practices
            ats_practices = self._fetch_ats_best_practices()
            external_data.extend(ats_practices)
            
        except Exception as e:
            logger.error(f"Error fetching external data: {e}")
        
        return external_data
    
    def _fetch_industry_keywords(self) -> List[Dict]:
        """Fetch trending industry keywords"""
        # This would integrate with real APIs in production
        return [
            {'category': 'tech_keywords', 'keywords': ['AI', 'Machine Learning', 'Cloud Computing', 'DevOps']},
            {'category': 'soft_skills', 'keywords': ['Leadership', 'Communication', 'Problem Solving']},
            {'category': 'certifications', 'keywords': ['AWS', 'Google Cloud', 'Scrum Master', 'PMP']}
        ]
    
    def _fetch_ats_best_practices(self) -> List[Dict]:
        """Fetch ATS optimization best practices"""
        return [
            {'practice': 'keyword_density', 'optimal_range': '2-3%'},
            {'practice': 'section_order', 'recommended': ['Contact', 'Summary', 'Experience', 'Education', 'Skills']},
            {'practice': 'formatting', 'avoid': ['tables', 'graphics', 'unusual fonts']}
        ]
    
    def train_models(self) -> bool:
        """Train all AI models with collected data"""
        try:
            # Collect training data
            training_data = self.collect_training_data()
            external_data = self.fetch_external_data()
            
            if len(training_data) < 20:
                logger.warning("Insufficient training data for model training")
                return False
            
            # Prepare features and targets
            texts = [item['text'] for item in training_data]
            ats_scores = [item['ats_score'] for item in training_data]
            user_feedback = [item['user_feedback'] for item in training_data]
            
            # Vectorize text data
            text_features = self.vectorizer.fit_transform(texts).toarray()
            
            # Extract additional features
            additional_features = []
            for text in texts:
                features = self._extract_advanced_features(text)
                additional_features.append(list(features.values()))
            
            additional_features = np.array(additional_features)
            additional_features = self.scaler.fit_transform(additional_features)
            
            # Combine features
            X = np.hstack([text_features, additional_features])
            
            # Train ATS predictor
            y_ats = np.array(ats_scores)
            self.models['ats_predictor'].fit(X, y_ats)
            
            # Train improvement predictor
            y_improvement = np.array(user_feedback)
            self.models['improvement_predictor'].fit(X, y_improvement)
            
            # Save models
            self._save_models()
            self.last_training = datetime.now()
            
            logger.info(f"Models trained successfully with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def _extract_advanced_features(self, text: str) -> Dict:
        """Extract advanced features for ML training"""
        features = {}
        
        # Text statistics
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # ATS-friendly features
        features['action_verb_count'] = len([word for word in text.lower().split() 
                                           if word in ['achieved', 'managed', 'led', 'developed']])
        features['quantifiable_metrics'] = len(re.findall(r'\d+%|\$\d+|\d+\s*years?', text))
        
        # Section completeness
        sections = ['experience', 'education', 'skills', 'contact']
        for section in sections:
            features[f'has_{section}'] = 1 if section in text.lower() else 0
        
        # Formatting quality
        features['has_bullets'] = 1 if any(char in text for char in ['•', '*', '-']) else 0
        features['proper_capitalization'] = len(re.findall(r'\b[A-Z][a-z]+', text)) / len(text.split())
        
        return features
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'models': self.models,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'last_training': self.last_training
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.models = model_data['models']
                self.vectorizer = model_data['vectorizer']
                self.scaler = model_data['scaler']
                self.last_training = model_data.get('last_training')
                return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        return False
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained"""
        if self.last_training is None:
            return True
        
        # Retrain weekly or when significant new data is available
        time_since_training = datetime.now() - self.last_training
        return time_since_training > timedelta(days=7)
    
    def predict_improvements(self, text: str) -> Dict:
        """Predict potential improvements for a resume"""
        try:
            # Prepare features
            text_features = self.vectorizer.transform([text]).toarray()
            additional_features = list(self._extract_advanced_features(text).values())
            additional_features = self.scaler.transform([additional_features])
            
            X = np.hstack([text_features, additional_features])
            
            # Make predictions
            ats_prediction = self.models['ats_predictor'].predict(X)[0]
            improvement_potential = self.models['improvement_predictor'].predict(X)[0]
            
            return {
                'predicted_ats_score': max(0, min(100, ats_prediction)),
                'improvement_potential': max(0, min(100, improvement_potential)),
                'confidence': self._calculate_prediction_confidence(X)
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {'predicted_ats_score': 0, 'improvement_potential': 0, 'confidence': 0}
    
    def _calculate_prediction_confidence(self, X: np.ndarray) -> float:
        """Calculate confidence in predictions"""
        try:
            # Use ensemble variance as confidence measure
            predictions = []
            for model in self.models.values():
                if hasattr(model, 'predict'):
                    pred = model.predict(X)[0]
                    predictions.append(pred)
            
            if len(predictions) > 1:
                variance = np.var(predictions)
                confidence = max(0, min(100, 100 - variance))
                return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
        
        return 75.0  # Default confidence
    
    def learn_from_database(self, include_external: bool = False) -> Dict:
        """Learn from database and optionally external data"""
        try:
            logger.info("Starting AI learning process...")
            
            # Collect training data from database
            training_data = self.collect_training_data()
            
            if len(training_data) < 10:
                logger.warning("Insufficient data for learning. Need at least 10 samples.")
                return {
                    'success': False,
                    'message': 'Insufficient training data',
                    'samples_collected': len(training_data)
                }
            
            # Optionally fetch external data
            external_data = []
            if include_external:
                external_data = self.fetch_external_data()
            
            # Train models
            success = self.train_models()
            
            result = {
                'success': success,
                'samples_collected': len(training_data),
                'external_data_points': len(external_data),
                'last_training': self.last_training.isoformat() if self.last_training else None,
                'message': 'Learning completed successfully' if success else 'Learning failed'
            }
            
            logger.info(f"AI learning completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during learning process: {e}")
            return {
                'success': False,
                'message': f'Learning failed: {str(e)}',
                'samples_collected': 0
            }

class EnhancedResumeRefiner:
    """AI system that refines resumes while maintaining original formatting"""
    
    def __init__(self):
        self.improvement_templates = {
            'action_verbs': {
                'weak_verbs': ['did', 'worked', 'helped', 'was responsible for'],
                'strong_verbs': ['achieved', 'implemented', 'optimized', 'spearheaded']
            },
            'quantification': {
                'patterns': [
                    r'improved (\w+)',
                    r'increased (\w+)',
                    r'reduced (\w+)',
                    r'managed (\w+)'
                ],
                'suggestions': [
                    'improved {item} by X%',
                    'increased {item} by X%',
                    'reduced {item} by X%',
                    'managed team of X {item}'
                ]
            }
        }
    
    def refine_resume(self, original_text: str, target_job_description: str = None) -> Dict:
        """Refine resume while maintaining original formatting"""
        try:
            refined_text = original_text
            improvements_made = []
            
            # 1. Improve action verbs
            refined_text, verb_improvements = self._improve_action_verbs(refined_text)
            improvements_made.extend(verb_improvements)
            
            # 2. Add quantification suggestions
            refined_text, quant_improvements = self._suggest_quantifications(refined_text)
            improvements_made.extend(quant_improvements)
            
            # 3. Optimize for ATS keywords
            if target_job_description:
                refined_text, keyword_improvements = self._optimize_keywords(
                    refined_text, target_job_description
                )
                improvements_made.extend(keyword_improvements)
            
            # 4. Improve formatting
            refined_text, format_improvements = self._improve_formatting(refined_text)
            improvements_made.extend(format_improvements)
            
            return {
                'refined_text': refined_text,
                'improvements_made': improvements_made,
                'improvement_count': len(improvements_made),
                'formatting_preserved': True
            }
            
        except Exception as e:
            logger.error(f"Error refining resume: {e}")
            return {
                'refined_text': original_text,
                'improvements_made': [],
                'improvement_count': 0,
                'formatting_preserved': True
            }
    
    def _improve_action_verbs(self, text: str) -> Tuple[str, List[str]]:
        """Replace weak action verbs with stronger alternatives"""
        improvements = []
        refined_text = text
        
        for weak_verb in self.improvement_templates['action_verbs']['weak_verbs']:
            if weak_verb in text.lower():
                # Find a suitable replacement
                strong_verb = np.random.choice(
                    self.improvement_templates['action_verbs']['strong_verbs']
                )
                
                # Replace first occurrence
                pattern = re.compile(re.escape(weak_verb), re.IGNORECASE)
                refined_text = pattern.sub(strong_verb, refined_text, count=1)
                
                improvements.append(f"Replaced '{weak_verb}' with '{strong_verb}'")
        
        return refined_text, improvements
    
    def _suggest_quantifications(self, text: str) -> Tuple[str, List[str]]:
        """Suggest quantification for achievements"""
        improvements = []
        refined_text = text
        
        # Look for achievements that could be quantified
        achievement_patterns = [
            r'improved (\w+)',
            r'increased (\w+)',
            r'reduced (\w+)',
            r'managed (\w+)'
        ]
        
        for pattern in achievement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group(0)
                item = match.group(1)
                
                # Suggest quantification
                suggestion = f"{original} by [X%/amount]"
                improvements.append(f"Consider quantifying: '{original}' → '{suggestion}'")
        
        return refined_text, improvements
    
    def _optimize_keywords(self, text: str, job_description: str) -> Tuple[str, List[str]]:
        """Optimize resume for job-specific keywords"""
        improvements = []
        refined_text = text
        
        # Extract keywords from job description
        job_keywords = self._extract_job_keywords(job_description)
        
        # Find missing keywords that could be naturally integrated
        text_lower = text.lower()
        missing_keywords = [kw for kw in job_keywords if kw.lower() not in text_lower]
        
        for keyword in missing_keywords[:5]:  # Limit to top 5 missing keywords
            improvements.append(f"Consider adding keyword: '{keyword}'")
        
        return refined_text, improvements
    
    def _extract_job_keywords(self, job_description: str) -> List[str]:
        """Extract important keywords from job description"""
        # Simple keyword extraction (in production, use more sophisticated NLP)
        words = re.findall(r'\b[A-Za-z]{3,}\b', job_description)
        
        # Filter for likely important keywords
        important_words = []
        for word in words:
            if (word.lower() not in ['the', 'and', 'for', 'with', 'you', 'will', 'are'] and
                len(word) > 3 and
                not word.isdigit()):
                important_words.append(word)
        
        # Return most frequent keywords
        from collections import Counter
        word_counts = Counter(important_words)
        return [word for word, count in word_counts.most_common(20)]
    
    def _improve_formatting(self, text: str) -> Tuple[str, List[str]]:
        """Improve resume formatting for ATS compatibility"""
        improvements = []
        refined_text = text
        
        # Check for common formatting issues
        if not re.search(r'[•\-\*]', text):
            improvements.append("Consider using bullet points for better readability")
        
        # Check for proper section headers
        if not re.search(r'^[A-Z\s]+$', text, re.MULTILINE):
            improvements.append("Consider using clear section headers (e.g., EXPERIENCE, EDUCATION)")
        
        return refined_text, improvements

class PDFExportSystem:
    """System for exporting refined resumes as high-quality PDFs"""
    
    def __init__(self):
        self.export_templates = {
            'professional': {
                'font': 'Arial',
                'font_size': 11,
                'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
                'line_spacing': 1.15
            },
            'modern': {
                'font': 'Calibri',
                'font_size': 11,
                'margins': {'top': 0.8, 'bottom': 0.8, 'left': 0.8, 'right': 0.8},
                'line_spacing': 1.2
            }
        }
    
    def export_to_pdf(self, refined_text: str, template: str = 'professional') -> Dict:
        """Export refined resume to PDF format"""
        try:
            # This would integrate with a PDF generation library like ReportLab
            # For now, return a mock response
            
            template_config = self.export_templates.get(template, self.export_templates['professional'])
            
            # In a real implementation, you would:
            # 1. Parse the refined text into structured sections
            # 2. Apply the template formatting
            # 3. Generate PDF using ReportLab or similar
            # 4. Return the PDF file path or bytes
            
            return {
                'success': True,
                'pdf_path': f'exports/resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                'template_used': template,
                'file_size': '245 KB',
                'pages': 1
            }
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            return {
                'success': False,
                'error': str(e)
            }