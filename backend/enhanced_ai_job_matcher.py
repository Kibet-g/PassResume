"""
Enhanced AI-Powered Job Matching System
Advanced machine learning algorithms for intelligent job recommendations
"""

import re
import json
import requests
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
import random
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobOpportunity:
    """Enhanced data class for job opportunities with AI scoring"""
    title: str
    company: str
    location: str
    description: str
    requirements: str
    salary_range: str
    job_type: str
    experience_level: str
    posted_date: str
    application_url: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    ai_confidence: float
    growth_potential: str
    culture_fit: float
    salary_prediction: str

@dataclass
class UserProfile:
    """Enhanced user profile with AI-extracted insights"""
    name: str
    email: str
    phone: str
    location: str
    skills: List[str]
    experience_years: int
    education_level: str
    job_titles: List[str]
    industries: List[str]
    keywords: List[str]
    career_level: str
    preferred_roles: List[str]
    salary_expectation: str
    work_preferences: Dict[str, Any]
    personality_traits: List[str]

class EnhancedAIJobMatcher:
    """
    Advanced AI-powered job matching system with:
    1. Deep learning for resume analysis
    2. Intelligent job compatibility scoring
    3. Predictive career path analysis
    4. Automated skill gap identification
    5. Personalized growth recommendations
    """
    
    def __init__(self):
        self.setup_ai_models()
        self.setup_job_sources()
        self.setup_enhanced_skill_database()
        self.setup_ml_models()
        
    def setup_ai_models(self):
        """Initialize advanced AI models"""
        try:
            # Load pre-trained language models
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize BERT-based models for semantic understanding
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                self.text_classifier = pipeline("zero-shot-classification")
                logger.info("Advanced AI models loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load transformer models: {e}")
                self.sentiment_analyzer = None
                self.text_classifier = None
                
        except OSError:
            logger.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None
            self.sentiment_analyzer = None
            self.text_classifier = None
            
        # Enhanced vectorizers for better text analysis
        self.skill_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        self.job_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def setup_ml_models(self):
        """Initialize machine learning models for predictions"""
        # Job compatibility classifier
        self.compatibility_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Neural network for advanced matching
        self.neural_matcher = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
        
        # Initialize with some training data
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate synthetic training data for job compatibility
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [skill_match_ratio, experience_match, education_match, location_match, salary_match]
        X = np.random.rand(n_samples, 5)
        
        # Target: compatibility score (0-1)
        y = (X[:, 0] * 0.4 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + X[:, 3] * 0.05 + X[:, 4] * 0.05 + 
             np.random.normal(0, 0.1, n_samples)).clip(0, 1)
        
        # Train models
        self.compatibility_model.fit(X, y)
        self.neural_matcher.fit(X, y)
        
        logger.info("ML models initialized with synthetic data")
        
    def setup_job_sources(self):
        """Configure enhanced job search APIs and sources"""
        self.job_sources = {
            'indeed': {
                'base_url': 'https://indeed.com/jobs',
                'enabled': True,
                'weight': 0.3
            },
            'linkedin': {
                'base_url': 'https://linkedin.com/jobs/search',
                'enabled': True,
                'weight': 0.4
            },
            'glassdoor': {
                'base_url': 'https://glassdoor.com/Job/jobs.htm',
                'enabled': True,
                'weight': 0.2
            },
            'remote_ok': {
                'base_url': 'https://remoteok.io/api',
                'enabled': True,
                'weight': 0.1
            }
        }
        
    def setup_enhanced_skill_database(self):
        """Initialize comprehensive skill database with AI categorization"""
        self.skill_categories = {
            'programming': {
                'skills': [
                    'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                    'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql', 'html',
                    'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
                    'express', 'laravel', 'rails', 'asp.net', 'jquery', 'bootstrap'
                ],
                'weight': 0.35,
                'growth_rate': 'high'
            },
            'data_science': {
                'skills': [
                    'machine learning', 'deep learning', 'data analysis', 'statistics',
                    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                    'tableau', 'power bi', 'excel', 'r', 'sas', 'spss', 'hadoop', 'spark',
                    'data mining', 'predictive modeling', 'neural networks', 'nlp'
                ],
                'weight': 0.4,
                'growth_rate': 'very_high'
            },
            'cloud': {
                'skills': [
                    'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform',
                    'jenkins', 'ci/cd', 'devops', 'microservices', 'serverless',
                    'cloudformation', 'ansible', 'chef', 'puppet'
                ],
                'weight': 0.3,
                'growth_rate': 'very_high'
            },
            'design': {
                'skills': [
                    'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd', 'ui/ux',
                    'graphic design', 'web design', 'user experience', 'user interface',
                    'prototyping', 'wireframing', 'user research', 'design thinking'
                ],
                'weight': 0.25,
                'growth_rate': 'medium'
            },
            'business': {
                'skills': [
                    'project management', 'agile', 'scrum', 'leadership', 'strategy',
                    'marketing', 'sales', 'finance', 'accounting', 'operations',
                    'business analysis', 'product management', 'stakeholder management'
                ],
                'weight': 0.2,
                'growth_rate': 'medium'
            },
            'soft_skills': {
                'skills': [
                    'communication', 'teamwork', 'problem solving', 'critical thinking',
                    'leadership', 'time management', 'adaptability', 'creativity',
                    'emotional intelligence', 'conflict resolution', 'negotiation'
                ],
                'weight': 0.15,
                'growth_rate': 'stable'
            }
        }
        
        # Create skill importance mapping
        self.skill_importance = {}
        for category, data in self.skill_categories.items():
            for skill in data['skills']:
                self.skill_importance[skill] = data['weight']
        
        # Flatten all skills for easy searching
        self.all_skills = []
        for category, data in self.skill_categories.items():
            self.all_skills.extend(data['skills'])
            
    def extract_enhanced_user_profile(self, resume_text: str) -> UserProfile:
        """Extract comprehensive user profile using AI analysis"""
        logger.info("Extracting enhanced user profile with AI analysis")
        
        # Basic extraction
        basic_profile = self._extract_basic_info(resume_text)
        
        # AI-enhanced extraction
        ai_insights = self._extract_ai_insights(resume_text)
        
        # Combine basic and AI insights
        return UserProfile(
            **basic_profile,
            **ai_insights
        )
        
    def _extract_basic_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract basic information from resume"""
        return {
            'name': self._extract_name(resume_text),
            'email': self._extract_email(resume_text),
            'phone': self._extract_phone(resume_text),
            'location': self._extract_location(resume_text),
            'skills': self._extract_skills_ai(resume_text),
            'experience_years': self._extract_experience_years(resume_text),
            'education_level': self._extract_education_level(resume_text),
            'job_titles': self._extract_job_titles(resume_text),
            'industries': self._extract_industries(resume_text),
            'keywords': self._extract_keywords_ai(resume_text)
        }
        
    def _extract_ai_insights(self, resume_text: str) -> Dict[str, Any]:
        """Extract AI-powered insights from resume"""
        insights = {
            'career_level': self._determine_career_level(resume_text),
            'preferred_roles': self._predict_preferred_roles(resume_text),
            'salary_expectation': self._estimate_salary_expectation(resume_text),
            'work_preferences': self._analyze_work_preferences(resume_text),
            'personality_traits': self._extract_personality_traits(resume_text)
        }
        
        return insights
        
    def _extract_skills_ai(self, text: str) -> List[str]:
        """AI-enhanced skill extraction with context understanding"""
        text_lower = text.lower()
        found_skills = []
        
        # Basic skill matching
        for skill in self.all_skills:
            if skill.lower() in text_lower:
                # Check context to ensure it's actually a skill mention
                if self._validate_skill_context(text_lower, skill.lower()):
                    found_skills.append(skill)
        
        # Use AI to find additional skills
        if self.text_classifier:
            try:
                # Classify text sections to find skill-related content
                skill_sections = self._identify_skill_sections(text)
                for section in skill_sections:
                    additional_skills = self._extract_skills_from_section(section)
                    found_skills.extend(additional_skills)
            except Exception as e:
                logger.warning(f"AI skill extraction failed: {e}")
        
        # Remove duplicates and sort by importance
        unique_skills = list(set(found_skills))
        return sorted(unique_skills, key=lambda x: self.skill_importance.get(x, 0), reverse=True)
        
    def _validate_skill_context(self, text: str, skill: str) -> bool:
        """Validate that a skill mention is in proper context"""
        # Find the skill in text and check surrounding words
        skill_index = text.find(skill)
        if skill_index == -1:
            return False
            
        # Get context around the skill (50 characters before and after)
        start = max(0, skill_index - 50)
        end = min(len(text), skill_index + len(skill) + 50)
        context = text[start:end]
        
        # Positive indicators
        positive_indicators = [
            'experience', 'proficient', 'skilled', 'expert', 'knowledge',
            'familiar', 'worked with', 'used', 'developed', 'implemented'
        ]
        
        # Negative indicators
        negative_indicators = [
            'not', 'no experience', 'unfamiliar', 'learning', 'beginner'
        ]
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in context)
        negative_score = sum(1 for indicator in negative_indicators if indicator in context)
        
        return positive_score > negative_score
        
    def _determine_career_level(self, text: str) -> str:
        """Determine career level using AI analysis"""
        experience_years = self._extract_experience_years(text)
        
        # Analyze job titles and responsibilities
        text_lower = text.lower()
        
        senior_indicators = ['senior', 'lead', 'principal', 'architect', 'manager', 'director']
        mid_indicators = ['developer', 'analyst', 'engineer', 'specialist']
        junior_indicators = ['junior', 'intern', 'trainee', 'entry', 'associate']
        
        senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
        mid_count = sum(1 for indicator in mid_indicators if indicator in text_lower)
        junior_count = sum(1 for indicator in junior_indicators if indicator in text_lower)
        
        if experience_years >= 8 or senior_count >= 2:
            return 'Senior'
        elif experience_years >= 3 or mid_count >= 2:
            return 'Mid-level'
        else:
            return 'Entry-level'
            
    def _predict_preferred_roles(self, text: str) -> List[str]:
        """Predict preferred job roles based on resume content"""
        skills = self._extract_skills_ai(text)
        job_titles = self._extract_job_titles(text)
        
        # Role prediction based on skill clusters
        role_predictions = []
        
        # Programming roles
        prog_skills = [s for s in skills if s in self.skill_categories['programming']['skills']]
        if len(prog_skills) >= 3:
            role_predictions.extend(['Software Engineer', 'Full Stack Developer', 'Backend Developer'])
            
        # Data science roles
        ds_skills = [s for s in skills if s in self.skill_categories['data_science']['skills']]
        if len(ds_skills) >= 3:
            role_predictions.extend(['Data Scientist', 'Machine Learning Engineer', 'Data Analyst'])
            
        # Cloud/DevOps roles
        cloud_skills = [s for s in skills if s in self.skill_categories['cloud']['skills']]
        if len(cloud_skills) >= 2:
            role_predictions.extend(['DevOps Engineer', 'Cloud Architect', 'Site Reliability Engineer'])
            
        # Design roles
        design_skills = [s for s in skills if s in self.skill_categories['design']['skills']]
        if len(design_skills) >= 2:
            role_predictions.extend(['UX Designer', 'Product Designer', 'UI Developer'])
            
        return list(set(role_predictions))[:5]  # Return top 5 unique predictions
        
    def _estimate_salary_expectation(self, text: str) -> str:
        """Estimate salary expectation based on skills and experience"""
        experience_years = self._extract_experience_years(text)
        skills = self._extract_skills_ai(text)
        career_level = self._determine_career_level(text)
        
        # Base salary by experience
        base_salary = {
            'Entry-level': 50000,
            'Mid-level': 80000,
            'Senior': 120000
        }.get(career_level, 50000)
        
        # Skill multipliers
        high_value_skills = ['machine learning', 'aws', 'kubernetes', 'react', 'python']
        skill_bonus = sum(5000 for skill in skills if skill in high_value_skills)
        
        estimated_salary = base_salary + skill_bonus + (experience_years * 3000)
        
        # Return salary range
        lower_bound = int(estimated_salary * 0.9)
        upper_bound = int(estimated_salary * 1.2)
        
        return f"${lower_bound:,} - ${upper_bound:,}"
        
    def _analyze_work_preferences(self, text: str) -> Dict[str, Any]:
        """Analyze work preferences from resume content"""
        text_lower = text.lower()
        
        preferences = {
            'remote_friendly': any(term in text_lower for term in ['remote', 'distributed', 'virtual']),
            'startup_experience': any(term in text_lower for term in ['startup', 'early-stage', 'founding']),
            'leadership_interest': any(term in text_lower for term in ['lead', 'manage', 'mentor', 'team lead']),
            'technical_focus': len([s for s in self._extract_skills_ai(text) 
                                  if s in self.skill_categories['programming']['skills']]) > 5,
            'collaboration_emphasis': any(term in text_lower for term in ['team', 'collaborate', 'cross-functional'])
        }
        
        return preferences
        
    def _extract_personality_traits(self, text: str) -> List[str]:
        """Extract personality traits from resume language"""
        if self.sentiment_analyzer:
            try:
                # Analyze sentiment and language patterns
                sentiment = self.sentiment_analyzer(text[:512])  # Limit text length
                
                traits = []
                if sentiment[0]['label'] == 'POSITIVE':
                    traits.extend(['optimistic', 'enthusiastic'])
                    
                # Analyze specific language patterns
                text_lower = text.lower()
                
                if any(term in text_lower for term in ['innovative', 'creative', 'design']):
                    traits.append('creative')
                if any(term in text_lower for term in ['analytical', 'data', 'research']):
                    traits.append('analytical')
                if any(term in text_lower for term in ['leadership', 'manage', 'lead']):
                    traits.append('leadership-oriented')
                if any(term in text_lower for term in ['detail', 'precise', 'accurate']):
                    traits.append('detail-oriented')
                    
                return traits[:5]  # Return top 5 traits
            except Exception as e:
                logger.warning(f"Personality trait extraction failed: {e}")
                
        return ['adaptable', 'problem-solver']  # Default traits
        
    def intelligent_job_matching(self, user_profile: UserProfile, jobs: List[Dict[str, Any]]) -> List[JobOpportunity]:
        """Advanced AI-powered job matching with multiple algorithms"""
        logger.info(f"Running intelligent job matching for {len(jobs)} jobs")
        
        matched_jobs = []
        
        for job in jobs:
            # Calculate multiple match scores
            semantic_score = self._calculate_semantic_similarity(user_profile, job)
            skill_score = self._calculate_skill_match_score(user_profile, job)
            experience_score = self._calculate_experience_match(user_profile, job)
            culture_score = self._calculate_culture_fit(user_profile, job)
            
            # Use ML model for final compatibility prediction
            features = np.array([[
                skill_score,
                experience_score,
                semantic_score,
                culture_score,
                0.5  # placeholder for location match
            ]])
            
            ml_score = self.compatibility_model.predict(features)[0]
            neural_score = self.neural_matcher.predict(features)[0]
            
            # Combine scores with weights
            final_score = (
                semantic_score * 0.25 +
                skill_score * 0.30 +
                experience_score * 0.20 +
                culture_score * 0.15 +
                ml_score * 0.05 +
                neural_score * 0.05
            ) * 100
            
            # Create enhanced job opportunity
            job_opportunity = self._create_enhanced_job_opportunity(
                job, user_profile, final_score, skill_score, culture_score
            )
            
            matched_jobs.append(job_opportunity)
            
        # Sort by match score and return top matches
        matched_jobs.sort(key=lambda x: x.match_score, reverse=True)
        return matched_jobs
        
    def _calculate_semantic_similarity(self, user_profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calculate semantic similarity between user profile and job"""
        try:
            # Combine user profile text
            user_text = f"{' '.join(user_profile.skills)} {' '.join(user_profile.keywords)} {' '.join(user_profile.job_titles)}"
            
            # Combine job text
            job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}"
            
            # Calculate TF-IDF similarity
            texts = [user_text, job_text]
            tfidf_matrix = self.job_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
            
    def _calculate_skill_match_score(self, user_profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calculate skill match score with importance weighting"""
        job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}".lower()
        
        user_skills = [skill.lower() for skill in user_profile.skills]
        
        matching_skills = []
        total_importance = 0
        matched_importance = 0
        
        for skill in self.all_skills:
            skill_lower = skill.lower()
            importance = self.skill_importance.get(skill, 0.1)
            total_importance += importance
            
            if skill_lower in job_text:
                if skill_lower in user_skills:
                    matching_skills.append(skill)
                    matched_importance += importance
                    
        return matched_importance / total_importance if total_importance > 0 else 0
        
    def _calculate_experience_match(self, user_profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calculate experience level match"""
        job_text = job.get('description', '').lower()
        
        # Extract required experience from job
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        required_years = 0
        for pattern in exp_patterns:
            matches = re.findall(pattern, job_text)
            if matches:
                required_years = max([int(match) for match in matches])
                break
                
        if required_years == 0:
            return 0.8  # Neutral score if no experience requirement found
            
        # Calculate match based on user's experience
        user_years = user_profile.experience_years
        
        if user_years >= required_years:
            # Bonus for having more experience, but diminishing returns
            excess = user_years - required_years
            return min(1.0, 0.9 + (excess * 0.02))
        else:
            # Penalty for having less experience
            deficit = required_years - user_years
            return max(0.1, 0.8 - (deficit * 0.1))
            
    def _calculate_culture_fit(self, user_profile: UserProfile, job: Dict[str, Any]) -> float:
        """Calculate culture fit based on work preferences and company description"""
        job_text = job.get('description', '').lower()
        company = job.get('company', '').lower()
        
        fit_score = 0.5  # Base score
        
        # Remote work preference
        if user_profile.work_preferences.get('remote_friendly', False):
            if any(term in job_text for term in ['remote', 'distributed', 'work from home']):
                fit_score += 0.2
                
        # Startup experience
        if user_profile.work_preferences.get('startup_experience', False):
            if any(term in job_text + company for term in ['startup', 'early-stage', 'fast-paced']):
                fit_score += 0.15
                
        # Leadership interest
        if user_profile.work_preferences.get('leadership_interest', False):
            if any(term in job_text for term in ['lead', 'mentor', 'manage', 'leadership']):
                fit_score += 0.15
                
        return min(1.0, fit_score)
        
    def _create_enhanced_job_opportunity(self, job: Dict[str, Any], user_profile: UserProfile, 
                                       match_score: float, skill_score: float, culture_score: float) -> JobOpportunity:
        """Create enhanced job opportunity with AI insights"""
        
        # Identify matching and missing skills
        job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}".lower()
        user_skills_lower = [skill.lower() for skill in user_profile.skills]
        
        matching_skills = [skill for skill in user_profile.skills if skill.lower() in job_text]
        
        # Find missing skills that are mentioned in the job
        missing_skills = []
        for skill in self.all_skills:
            if skill.lower() in job_text and skill.lower() not in user_skills_lower:
                missing_skills.append(skill)
                
        # Predict growth potential
        growth_potential = self._predict_growth_potential(job, user_profile)
        
        # Predict salary
        salary_prediction = self._predict_job_salary(job, user_profile)
        
        return JobOpportunity(
            title=job.get('title', 'Unknown'),
            company=job.get('company', 'Unknown'),
            location=job.get('location', 'Unknown'),
            description=job.get('description', ''),
            requirements=job.get('requirements', ''),
            salary_range=job.get('salary_range', 'Not specified'),
            job_type=job.get('job_type', 'Full-time'),
            experience_level=job.get('experience_level', 'Mid-level'),
            posted_date=job.get('posted_date', datetime.now().strftime('%Y-%m-%d')),
            application_url=job.get('application_url', ''),
            match_score=round(match_score, 1),
            matching_skills=matching_skills[:10],  # Top 10 matching skills
            missing_skills=missing_skills[:5],     # Top 5 missing skills
            ai_confidence=round(skill_score * 100, 1),
            growth_potential=growth_potential,
            culture_fit=round(culture_score * 100, 1),
            salary_prediction=salary_prediction
        )
        
    def _predict_growth_potential(self, job: Dict[str, Any], user_profile: UserProfile) -> str:
        """Predict career growth potential for this job"""
        job_text = job.get('description', '').lower()
        title = job.get('title', '').lower()
        
        # High growth indicators
        high_growth_terms = [
            'machine learning', 'ai', 'cloud', 'kubernetes', 'microservices',
            'leadership', 'senior', 'principal', 'architect', 'lead'
        ]
        
        # Medium growth indicators
        medium_growth_terms = [
            'full stack', 'react', 'python', 'javascript', 'data',
            'product', 'agile', 'scrum'
        ]
        
        high_count = sum(1 for term in high_growth_terms if term in job_text + title)
        medium_count = sum(1 for term in medium_growth_terms if term in job_text + title)
        
        if high_count >= 2:
            return 'High'
        elif medium_count >= 2 or high_count >= 1:
            return 'Medium'
        else:
            return 'Low'
            
    def _predict_job_salary(self, job: Dict[str, Any], user_profile: UserProfile) -> str:
        """Predict salary for this specific job"""
        # Use existing salary if provided
        if job.get('salary_range') and job.get('salary_range') != 'Not specified':
            return job.get('salary_range')
            
        # Predict based on title and location
        title = job.get('title', '').lower()
        location = job.get('location', '').lower()
        
        # Base salary by role type
        role_salaries = {
            'senior': 120000,
            'lead': 130000,
            'principal': 150000,
            'architect': 140000,
            'manager': 110000,
            'engineer': 90000,
            'developer': 85000,
            'analyst': 75000,
            'designer': 80000
        }
        
        base_salary = 70000  # Default
        for role, salary in role_salaries.items():
            if role in title:
                base_salary = salary
                break
                
        # Location adjustments
        if any(city in location for city in ['san francisco', 'new york', 'seattle']):
            base_salary *= 1.3
        elif any(city in location for city in ['austin', 'boston', 'chicago']):
            base_salary *= 1.1
        elif 'remote' in location:
            base_salary *= 1.05
            
        # Return range
        lower = int(base_salary * 0.9)
        upper = int(base_salary * 1.2)
        
        return f"${lower:,} - ${upper:,}"

    # Include all the original methods for basic extraction
    def _extract_name(self, text: str) -> str:
        """Extract name from resume text"""
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if len(line.split()) == 2 and line.replace(' ', '').isalpha():
                return line
        return "Unknown"
        
    def _extract_email(self, text: str) -> str:
        """Extract email from resume text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else ""
        
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from resume text"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        return ''.join(phones[0]) if phones else ""
        
    def _extract_location(self, text: str) -> str:
        """Extract location from resume text"""
        location_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]{2})',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2})',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
                
        return "Remote"
        
    def _extract_experience_years(self, text: str) -> int:
        """Extract years of experience from resume text"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience.*?(\d+)\+?\s*years?'
        ]
        
        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches])
            
        return max(years) if years else 0
        
    def _extract_education_level(self, text: str) -> str:
        """Extract education level from resume text"""
        text_lower = text.lower()
        
        if any(degree in text_lower for degree in ['phd', 'ph.d', 'doctorate']):
            return 'PhD'
        elif any(degree in text_lower for degree in ['master', 'mba', 'ms', 'm.s']):
            return 'Masters'
        elif any(degree in text_lower for degree in ['bachelor', 'bs', 'b.s', 'ba', 'b.a']):
            return 'Bachelors'
        elif any(degree in text_lower for degree in ['associate', 'diploma']):
            return 'Associate'
        else:
            return 'High School'
            
    def _extract_job_titles(self, text: str) -> List[str]:
        """Extract job titles from resume text"""
        title_patterns = [
            r'(?:senior|junior|lead|principal)?\s*(?:software|data|web|mobile)?\s*(?:engineer|developer|analyst|scientist|manager|director)',
            r'(?:product|project|program)\s*manager',
            r'(?:ui/ux|ux|ui)\s*designer',
            r'(?:full\s*stack|front\s*end|back\s*end)\s*developer'
        ]
        
        titles = []
        for pattern in title_patterns:
            matches = re.findall(pattern, text.lower())
            titles.extend(matches)
            
        return list(set(titles))
        
    def _extract_industries(self, text: str) -> List[str]:
        """Extract industries from resume text"""
        industries = [
            'technology', 'healthcare', 'finance', 'education', 'retail',
            'manufacturing', 'consulting', 'media', 'telecommunications',
            'automotive', 'aerospace', 'energy', 'real estate'
        ]
        
        text_lower = text.lower()
        found_industries = []
        
        for industry in industries:
            if industry in text_lower:
                found_industries.append(industry)
                
        return found_industries
        
    def _extract_keywords_ai(self, text: str) -> List[str]:
        """AI-enhanced keyword extraction"""
        if self.nlp:
            doc = self.nlp(text)
            keywords = [token.lemma_.lower() for token in doc 
                       if token.is_alpha and not token.is_stop and len(token.text) > 2]
        else:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if word not in stop_words]
            
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(30)]
        
    def _identify_skill_sections(self, text: str) -> List[str]:
        """Identify sections of text that likely contain skills"""
        sections = []
        lines = text.split('\n')
        
        skill_section_headers = [
            'skills', 'technical skills', 'technologies', 'tools',
            'programming languages', 'software', 'expertise'
        ]
        
        in_skill_section = False
        current_section = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a skill section header
            if any(header in line_lower for header in skill_section_headers):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
                in_skill_section = True
            elif in_skill_section:
                if line.strip() == '' or line_lower.startswith(('experience', 'education', 'work')):
                    # End of skill section
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = []
                    in_skill_section = False
                else:
                    current_section.append(line)
                    
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections
        
    def _extract_skills_from_section(self, section: str) -> List[str]:
        """Extract skills from a specific section of text"""
        section_lower = section.lower()
        found_skills = []
        
        for skill in self.all_skills:
            if skill.lower() in section_lower:
                found_skills.append(skill)
                
        return found_skills

    # Mock job generation and search methods (keeping original functionality)
    def search_jobs_online(self, user_profile: UserProfile, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for jobs online using multiple sources with AI enhancement"""
        logger.info(f"AI-enhanced job search for user in {user_profile.location}")
        
        all_jobs = []
        
        # Search RemoteOK (has API)
        remote_jobs = self._search_remote_ok_enhanced(user_profile, limit // 2)
        all_jobs.extend(remote_jobs)
        
        # Generate AI-enhanced mock jobs
        mock_jobs = self._generate_ai_enhanced_mock_jobs(user_profile, limit // 2)
        all_jobs.extend(mock_jobs)
        
        return all_jobs[:limit]
        
    def _search_remote_ok_enhanced(self, user_profile: UserProfile, limit: int) -> List[Dict[str, Any]]:
        """Enhanced RemoteOK search with AI filtering"""
        try:
            response = requests.get('https://remoteok.io/api', timeout=10)
            if response.status_code == 200:
                jobs_data = response.json()[1:]
                
                relevant_jobs = []
                for job in jobs_data[:limit * 2]:  # Get more jobs for better filtering
                    if self._is_job_relevant_ai(job, user_profile):
                        enhanced_job = self._enhance_job_data(job, user_profile)
                        relevant_jobs.append(enhanced_job)
                        
                return relevant_jobs[:limit]
        except Exception as e:
            logger.error(f"Error searching RemoteOK: {e}")
            
        return []
        
    def _is_job_relevant_ai(self, job: Dict, user_profile: UserProfile) -> bool:
        """AI-enhanced job relevance checking"""
        job_text = f"{job.get('position', '')} {job.get('description', '')} {' '.join(job.get('tags', []))}"
        
        # Calculate semantic similarity
        try:
            user_text = f"{' '.join(user_profile.skills)} {' '.join(user_profile.keywords[:10])}"
            texts = [user_text, job_text]
            tfidf_matrix = self.job_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity > 0.1  # Threshold for relevance
        except:
            # Fallback to basic matching
            job_text_lower = job_text.lower()
            skill_matches = sum(1 for skill in user_profile.skills if skill.lower() in job_text_lower)
            return skill_matches >= 2
            
    def _enhance_job_data(self, job: Dict, user_profile: UserProfile) -> Dict[str, Any]:
        """Enhance job data with AI predictions"""
        enhanced_job = {
            'title': job.get('position', 'Unknown'),
            'company': job.get('company', 'Unknown'),
            'location': 'Remote',
            'description': job.get('description', ''),
            'requirements': ', '.join(job.get('tags', [])),
            'salary_range': f"${job.get('salary_min', 0)}-${job.get('salary_max', 0)}" if job.get('salary_min') else 'Not specified',
            'job_type': 'Remote',
            'experience_level': self._predict_experience_level(job),
            'posted_date': datetime.now().strftime('%Y-%m-%d'),
            'application_url': job.get('url', ''),
            'source': 'RemoteOK'
        }
        
        return enhanced_job
        
    def _predict_experience_level(self, job: Dict) -> str:
        """Predict experience level required for job"""
        title = job.get('position', '').lower()
        description = job.get('description', '').lower()
        
        if any(term in title + description for term in ['senior', 'lead', 'principal', 'architect']):
            return 'Senior'
        elif any(term in title + description for term in ['junior', 'entry', 'intern']):
            return 'Entry-level'
        else:
            return 'Mid-level'
            
    def _generate_ai_enhanced_mock_jobs(self, user_profile: UserProfile, limit: int) -> List[Dict[str, Any]]:
        """Generate AI-enhanced mock job listings"""
        mock_companies = [
            'TechCorp', 'InnovateLabs', 'DataDriven Inc', 'CloudFirst Solutions',
            'AI Dynamics', 'NextGen Systems', 'Digital Pioneers', 'SmartTech',
            'FutureSoft', 'CodeCrafters', 'ByteBuilders', 'PixelPerfect'
        ]
        
        # Generate jobs based on user's preferred roles
        preferred_roles = user_profile.preferred_roles or ['Software Engineer', 'Developer']
        
        jobs = []
        for i in range(limit):
            role = random.choice(preferred_roles)
            company = random.choice(mock_companies)
            
            # Generate job based on user skills
            relevant_skills = random.sample(user_profile.skills, min(5, len(user_profile.skills)))
            
            job = {
                'title': role,
                'company': company,
                'location': random.choice(['Remote', 'New York, NY', 'San Francisco, CA', 'Austin, TX']),
                'description': f"Join {company} as a {role}. Work with cutting-edge technologies including {', '.join(relevant_skills[:3])}.",
                'requirements': f"Experience with {', '.join(relevant_skills)}, strong problem-solving skills",
                'salary_range': self._generate_salary_range(role, user_profile.career_level),
                'job_type': random.choice(['Full-time', 'Contract', 'Remote']),
                'experience_level': user_profile.career_level,
                'posted_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'application_url': f"https://{company.lower().replace(' ', '')}.com/careers",
                'source': 'AI Generated'
            }
            
            jobs.append(job)
            
        return jobs
        
    def _generate_salary_range(self, role: str, career_level: str) -> str:
        """Generate realistic salary range based on role and level"""
        base_salaries = {
            'Entry-level': {'Software Engineer': 70000, 'Data Analyst': 60000, 'UX Designer': 65000},
            'Mid-level': {'Software Engineer': 95000, 'Data Scientist': 110000, 'Product Manager': 120000},
            'Senior': {'Senior Software Engineer': 130000, 'Senior Data Scientist': 150000, 'Engineering Manager': 160000}
        }
        
        base = base_salaries.get(career_level, {}).get(role, 80000)
        lower = int(base * 0.9)
        upper = int(base * 1.3)
        
        return f"${lower:,} - ${upper:,}"