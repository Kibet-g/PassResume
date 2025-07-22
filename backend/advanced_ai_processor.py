"""
Advanced AI Resume Processor with Deep Analysis, Formatting Preservation, and Self-Training
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
import cv2
import base64
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import fitz  # PyMuPDF for PDF processing
from PIL import Image, ImageDraw, ImageFont
import io
import docx
from docx.shared import Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import spacy
import nltk
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)

class AdvancedAIResumeProcessor:
    """
    Advanced AI system that:
    1. Analyzes resumes in extreme detail
    2. Preserves original formatting and styling
    3. Makes intelligent improvements
    4. Continuously trains itself
    """
    
    def __init__(self):
        self.setup_ai_models()
        self.setup_formatting_analyzer()
        self.setup_self_training_system()
        self.performance_metrics = {
            'total_processed': 0,
            'improvement_success_rate': 0.0,
            'user_satisfaction_score': 0.0,
            'model_accuracy': 0.0
        }
        
    def setup_ai_models(self):
        """Initialize advanced AI models"""
        try:
            # Load spaCy model for NLP
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Initialize transformer models for advanced text analysis
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.text_classifier = pipeline("text-classification", 
                                           model="microsoft/DialoGPT-medium")
        except:
            logger.warning("Transformer models not available")
            self.sentiment_analyzer = None
            self.text_classifier = None
            
        # Custom ML models
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.improvement_model = RandomForestRegressor(n_estimators=100)
        self.formatting_model = GradientBoostingRegressor(n_estimators=100)
        self.content_quality_model = MLPRegressor(hidden_layer_sizes=(100, 50))
        
    def setup_formatting_analyzer(self):
        """Initialize formatting analysis capabilities"""
        self.formatting_patterns = {
            'fonts': {
                'professional': ['Arial', 'Calibri', 'Times New Roman', 'Helvetica'],
                'modern': ['Roboto', 'Open Sans', 'Lato', 'Montserrat'],
                'creative': ['Georgia', 'Garamond', 'Futura', 'Proxima Nova']
            },
            'colors': {
                'professional': ['#000000', '#333333', '#2C3E50', '#34495E'],
                'accent': ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'],
                'neutral': ['#7F8C8D', '#95A5A6', '#BDC3C7', '#ECF0F1']
            },
            'layouts': {
                'traditional': {'margins': 1.0, 'spacing': 1.15, 'sections': 'linear'},
                'modern': {'margins': 0.8, 'spacing': 1.0, 'sections': 'columnar'},
                'creative': {'margins': 0.5, 'spacing': 1.2, 'sections': 'mixed'}
            }
        }
        
    def setup_self_training_system(self):
        """Initialize self-training capabilities"""
        self.training_data = {
            'successful_improvements': [],
            'user_feedback': [],
            'performance_metrics': [],
            'content_patterns': [],
            'formatting_preferences': []
        }
        self.learning_threshold = 50  # Minimum samples before retraining
        self.last_training = None
        
    def analyze_resume_deeply(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Perform deep analysis of resume including:
        - Content analysis
        - Formatting analysis
        - Structure analysis
        - ATS compatibility
        - Improvement suggestions
        """
        analysis_result = {
            'content_analysis': {},
            'formatting_analysis': {},
            'structure_analysis': {},
            'ats_analysis': {},
            'improvement_suggestions': {},
            'preserved_formatting': {},
            'confidence_scores': {}
        }
        
        try:
            # Extract content and formatting
            if file_type.lower() == 'pdf':
                content, formatting = self._extract_pdf_content_and_formatting(file_path)
            elif file_type.lower() in ['docx', 'doc']:
                content, formatting = self._extract_docx_content_and_formatting(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Deep content analysis
            analysis_result['content_analysis'] = self._analyze_content_deeply(content)
            
            # Formatting analysis
            analysis_result['formatting_analysis'] = self._analyze_formatting(formatting)
            
            # Structure analysis
            analysis_result['structure_analysis'] = self._analyze_structure(content)
            
            # ATS compatibility analysis
            analysis_result['ats_analysis'] = self._analyze_ats_compatibility(content, formatting)
            
            # Generate improvement suggestions
            analysis_result['improvement_suggestions'] = self._generate_intelligent_improvements(
                content, formatting, analysis_result
            )
            
            # Preserve original formatting for reconstruction
            analysis_result['preserved_formatting'] = formatting
            
            # Calculate confidence scores
            analysis_result['confidence_scores'] = self._calculate_confidence_scores(analysis_result)
            
            # Update performance metrics
            self.performance_metrics['total_processed'] += 1
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            return {'error': str(e)}
    
    def _extract_pdf_content_and_formatting(self, file_path: str) -> Tuple[str, Dict]:
        """Extract both content and formatting information from PDF"""
        content = ""
        formatting = {
            'fonts': [],
            'colors': [],
            'layout': {},
            'images': [],
            'tables': [],
            'styles': {}
        }
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with formatting
                text_dict = page.get_text("dict")
                page_content, page_formatting = self._process_pdf_page(text_dict)
                content += page_content + "\n"
                
                # Merge formatting information
                formatting['fonts'].extend(page_formatting.get('fonts', []))
                formatting['colors'].extend(page_formatting.get('colors', []))
                formatting['layout'][f'page_{page_num}'] = page_formatting.get('layout', {})
                
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            
        return content, formatting
    
    def _process_pdf_page(self, text_dict: Dict) -> Tuple[str, Dict]:
        """Process a single PDF page to extract content and formatting"""
        content = ""
        formatting = {
            'fonts': [],
            'colors': [],
            'layout': {'blocks': []},
            'styles': {}
        }
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                block_info = {
                    'bbox': block['bbox'],
                    'lines': []
                }
                
                for line in block["lines"]:
                    line_content = ""
                    line_info = {
                        'bbox': line['bbox'],
                        'spans': []
                    }
                    
                    for span in line["spans"]:
                        text = span.get("text", "")
                        line_content += text
                        
                        # Extract formatting information
                        font_info = {
                            'font': span.get('font', ''),
                            'size': span.get('size', 0),
                            'flags': span.get('flags', 0),
                            'color': span.get('color', 0)
                        }
                        
                        formatting['fonts'].append(font_info)
                        line_info['spans'].append({
                            'text': text,
                            'formatting': font_info,
                            'bbox': span.get('bbox', [])
                        })
                    
                    content += line_content + "\n"
                    block_info['lines'].append(line_info)
                
                formatting['layout']['blocks'].append(block_info)
        
        return content, formatting
    
    def _extract_docx_content_and_formatting(self, file_path: str) -> Tuple[str, Dict]:
        """Extract both content and formatting information from DOCX"""
        content = ""
        formatting = {
            'fonts': [],
            'colors': [],
            'layout': {},
            'styles': {},
            'paragraphs': []
        }
        
        try:
            doc = docx.Document(file_path)
            
            for para in doc.paragraphs:
                para_content = para.text
                content += para_content + "\n"
                
                # Extract paragraph formatting
                para_formatting = {
                    'text': para_content,
                    'style': para.style.name if para.style else 'Normal',
                    'alignment': para.alignment,
                    'runs': []
                }
                
                for run in para.runs:
                    run_formatting = {
                        'text': run.text,
                        'bold': run.bold,
                        'italic': run.italic,
                        'underline': run.underline,
                        'font_name': run.font.name if run.font.name else 'Default',
                        'font_size': run.font.size.pt if run.font.size else 12,
                        'color': self._extract_color(run.font.color) if run.font.color else '#000000'
                    }
                    para_formatting['runs'].append(run_formatting)
                    formatting['fonts'].append(run_formatting)
                
                formatting['paragraphs'].append(para_formatting)
                
        except Exception as e:
            logger.error(f"Error extracting DOCX: {str(e)}")
            
        return content, formatting
    
    def _extract_color(self, color_obj) -> str:
        """Extract color information from docx color object"""
        try:
            if hasattr(color_obj, 'rgb') and color_obj.rgb:
                rgb = color_obj.rgb
                return f"#{rgb:06x}"
            return "#000000"
        except:
            return "#000000"
    
    def _analyze_content_deeply(self, content: str) -> Dict[str, Any]:
        """Perform deep content analysis using AI"""
        analysis = {
            'sections': {},
            'keywords': [],
            'sentiment': {},
            'readability': {},
            'achievements': [],
            'skills': [],
            'experience_quality': {},
            'language_quality': {}
        }
        
        try:
            # Section identification and analysis
            sections = self._identify_sections(content)
            for section_name, section_content in sections.items():
                analysis['sections'][section_name] = self._analyze_section_content(
                    section_name, section_content
                )
            
            # Keyword extraction
            analysis['keywords'] = self._extract_keywords_advanced(content)
            
            # Sentiment analysis
            if self.sentiment_analyzer:
                analysis['sentiment'] = self._analyze_sentiment(content)
            
            # Readability analysis
            analysis['readability'] = self._analyze_readability(content)
            
            # Achievement extraction
            analysis['achievements'] = self._extract_achievements(content)
            
            # Skills analysis
            analysis['skills'] = self._analyze_skills_advanced(content)
            
            # Experience quality assessment
            analysis['experience_quality'] = self._assess_experience_quality(content)
            
            # Language quality analysis
            analysis['language_quality'] = self._analyze_language_quality(content)
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            
        return analysis
    
    def _identify_sections(self, content: str) -> Dict[str, str]:
        """Identify and extract different sections of the resume"""
        sections = {}
        
        section_patterns = {
            'contact': [r'contact', r'personal\s+information', r'details'],
            'summary': [r'summary', r'profile', r'objective', r'about'],
            'experience': [r'experience', r'work\s+history', r'employment', r'career'],
            'education': [r'education', r'academic', r'qualifications', r'degree'],
            'skills': [r'skills', r'technical\s+skills', r'competencies', r'expertise'],
            'projects': [r'projects', r'portfolio', r'work\s+samples'],
            'certifications': [r'certifications', r'certificates', r'licenses'],
            'awards': [r'awards', r'honors', r'achievements', r'recognition']
        }
        
        content_lower = content.lower()
        lines = content.split('\n')
        
        for section_name, patterns in section_patterns.items():
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                for pattern in patterns:
                    if re.search(pattern, line_lower) and len(line_lower) < 50:
                        # Found section header, extract content
                        section_content = self._extract_section_content(lines, i)
                        sections[section_name] = section_content
                        break
                if section_name in sections:
                    break
        
        return sections
    
    def _extract_section_content(self, lines: List[str], start_index: int) -> str:
        """Extract content for a specific section"""
        content = []
        
        # Skip the header line
        for i in range(start_index + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop if we hit another section header
            if self._is_section_header(line):
                break
                
            if line:  # Skip empty lines
                content.append(line)
        
        return '\n'.join(content)
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header"""
        line_lower = line.lower()
        header_patterns = [
            r'^(contact|summary|profile|objective|experience|education|skills|projects|certifications|awards)',
            r'^[A-Z\s]{3,20}$',  # All caps short lines
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, line_lower):
                return True
        return False
    
    def improve_resume_intelligently(self, analysis_result: Dict[str, Any], 
                                   improvement_preferences: Dict = None) -> Dict[str, Any]:
        """
        Intelligently improve the resume while preserving formatting
        """
        improved_resume = {
            'improved_content': {},
            'formatting_enhancements': {},
            'ats_optimizations': {},
            'before_after_comparison': {},
            'improvement_confidence': {}
        }
        
        try:
            # Content improvements
            improved_resume['improved_content'] = self._improve_content(
                analysis_result, improvement_preferences
            )
            
            # Formatting enhancements
            improved_resume['formatting_enhancements'] = self._enhance_formatting(
                analysis_result['preserved_formatting'], improvement_preferences
            )
            
            # ATS optimizations
            improved_resume['ats_optimizations'] = self._optimize_for_ats(
                analysis_result
            )
            
            # Generate before/after comparison
            improved_resume['before_after_comparison'] = self._generate_comparison(
                analysis_result, improved_resume
            )
            
            # Calculate improvement confidence
            improved_resume['improvement_confidence'] = self._calculate_improvement_confidence(
                improved_resume
            )
            
            # Learn from this improvement
            self._learn_from_improvement(analysis_result, improved_resume)
            
        except Exception as e:
            logger.error(f"Error in intelligent improvement: {str(e)}")
            
        return improved_resume
    
    def _learn_from_improvement(self, analysis: Dict, improvement: Dict):
        """Learn from successful improvements for self-training"""
        learning_data = {
            'timestamp': datetime.now().isoformat(),
            'original_analysis': analysis,
            'improvements_made': improvement,
            'success_indicators': self._calculate_success_indicators(analysis, improvement)
        }
        
        self.training_data['successful_improvements'].append(learning_data)
        
        # Trigger retraining if we have enough data
        if len(self.training_data['successful_improvements']) >= self.learning_threshold:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain AI models based on accumulated learning data"""
        try:
            logger.info("Starting model retraining...")
            
            # Prepare training data
            training_features, training_targets = self._prepare_training_data()
            
            if len(training_features) > 10:  # Minimum samples for training
                # Retrain improvement model
                self.improvement_model.fit(training_features, training_targets)
                
                # Update performance metrics
                self.performance_metrics['model_accuracy'] = self._evaluate_model_performance()
                self.last_training = datetime.now()
                
                logger.info(f"Model retraining completed. New accuracy: {self.performance_metrics['model_accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
    
    def generate_formatted_resume(self, improved_content: Dict, original_formatting: Dict, 
                                output_format: str = 'pdf') -> bytes:
        """
        Generate a new resume file with improvements while preserving original formatting
        """
        try:
            if output_format.lower() == 'pdf':
                return self._generate_pdf_resume(improved_content, original_formatting)
            elif output_format.lower() == 'docx':
                return self._generate_docx_resume(improved_content, original_formatting)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error generating formatted resume: {str(e)}")
            return b""
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and performance metrics"""
        return {
            'performance_metrics': self.performance_metrics,
            'training_data_size': len(self.training_data['successful_improvements']),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'learning_threshold': self.learning_threshold,
            'models_loaded': {
                'nlp': self.nlp is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'improvement_model': hasattr(self.improvement_model, 'feature_importances_'),
                'formatting_model': hasattr(self.formatting_model, 'feature_importances_')
            }
        }
    
    # Additional helper methods would be implemented here...
    # (Due to length constraints, showing the main structure)