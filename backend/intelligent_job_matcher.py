"""
Intelligent Job Matching System
Analyzes resumes and matches users to relevant job opportunities online
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
import numpy as np
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobOpportunity:
    """Data class for job opportunities"""
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

@dataclass
class UserProfile:
    """Data class for user profile extracted from resume"""
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

class IntelligentJobMatcher:
    """
    Advanced job matching system that:
    1. Extracts comprehensive user profile from resume
    2. Searches multiple job platforms
    3. Analyzes job compatibility using AI
    4. Provides personalized recommendations
    """
    
    def __init__(self):
        self.setup_nlp_models()
        self.setup_job_sources()
        self.setup_skill_database()
        
    def setup_nlp_models(self):
        """Initialize NLP models for text analysis"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None
            
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def setup_job_sources(self):
        """Configure job search APIs and sources"""
        self.job_sources = {
            'indeed': {
                'base_url': 'https://indeed.com/jobs',
                'enabled': True
            },
            'linkedin': {
                'base_url': 'https://linkedin.com/jobs/search',
                'enabled': True
            },
            'glassdoor': {
                'base_url': 'https://glassdoor.com/Job/jobs.htm',
                'enabled': True
            },
            'remote_ok': {
                'base_url': 'https://remoteok.io/api',
                'enabled': True
            }
        }
        
    def setup_skill_database(self):
        """Initialize comprehensive skill database"""
        self.skill_categories = {
            'programming': [
                'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sql', 'html',
                'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                'tableau', 'power bi', 'excel', 'r', 'sas', 'spss', 'hadoop', 'spark'
            ],
            'cloud': [
                'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'ci/cd', 'devops', 'microservices', 'serverless'
            ],
            'design': [
                'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd', 'ui/ux',
                'graphic design', 'web design', 'user experience', 'user interface'
            ],
            'business': [
                'project management', 'agile', 'scrum', 'leadership', 'strategy',
                'marketing', 'sales', 'finance', 'accounting', 'operations'
            ],
            'soft_skills': [
                'communication', 'teamwork', 'problem solving', 'critical thinking',
                'leadership', 'time management', 'adaptability', 'creativity'
            ]
        }
        
        # Flatten all skills for easy searching
        self.all_skills = []
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)
            
    def extract_user_profile(self, resume_text: str) -> UserProfile:
        """Extract comprehensive user profile from resume text"""
        logger.info("Extracting user profile from resume")
        
        # Extract basic contact information
        name = self._extract_name(resume_text)
        email = self._extract_email(resume_text)
        phone = self._extract_phone(resume_text)
        location = self._extract_location(resume_text)
        
        # Extract skills
        skills = self._extract_skills(resume_text)
        
        # Extract experience
        experience_years = self._extract_experience_years(resume_text)
        
        # Extract education
        education_level = self._extract_education_level(resume_text)
        
        # Extract job titles and industries
        job_titles = self._extract_job_titles(resume_text)
        industries = self._extract_industries(resume_text)
        
        # Extract keywords
        keywords = self._extract_keywords(resume_text)
        
        return UserProfile(
            name=name,
            email=email,
            phone=phone,
            location=location,
            skills=skills,
            experience_years=experience_years,
            education_level=education_level,
            job_titles=job_titles,
            industries=industries,
            keywords=keywords
        )
        
    def _extract_name(self, text: str) -> str:
        """Extract name from resume text"""
        lines = text.split('\n')[:5]  # Check first 5 lines
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
        # Common location patterns
        location_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]{2})',  # City, State
            r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2})',  # City City, State
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',  # City, Country
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
                
        return "Remote"
        
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.all_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
                
        # Remove duplicates and return
        return list(set(found_skills))
        
    def _extract_experience_years(self, text: str) -> int:
        """Extract years of experience from resume text"""
        # Look for patterns like "5 years", "3+ years", etc.
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
        # Common job title patterns
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
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from resume text"""
        if self.nlp:
            doc = self.nlp(text)
            keywords = [token.lemma_.lower() for token in doc 
                       if token.is_alpha and not token.is_stop and len(token.text) > 2]
        else:
            # Basic keyword extraction without spaCy
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if word not in stop_words]
            
        # Get most frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]
        
    def search_jobs_online(self, user_profile: UserProfile, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for jobs online using multiple sources"""
        logger.info(f"Searching for jobs for user in {user_profile.location}")
        
        all_jobs = []
        
        # Search RemoteOK (has API)
        remote_jobs = self._search_remote_ok(user_profile, limit // 2)
        all_jobs.extend(remote_jobs)
        
        # Generate mock jobs for demonstration (in production, integrate real APIs)
        mock_jobs = self._generate_mock_jobs(user_profile, limit // 2)
        all_jobs.extend(mock_jobs)
        
        return all_jobs[:limit]
        
    def _search_remote_ok(self, user_profile: UserProfile, limit: int) -> List[Dict[str, Any]]:
        """Search RemoteOK API for remote jobs"""
        try:
            response = requests.get('https://remoteok.io/api', timeout=10)
            if response.status_code == 200:
                jobs_data = response.json()[1:]  # Skip first element (metadata)
                
                relevant_jobs = []
                for job in jobs_data[:limit]:
                    if self._is_job_relevant(job, user_profile):
                        relevant_jobs.append({
                            'title': job.get('position', 'Unknown'),
                            'company': job.get('company', 'Unknown'),
                            'location': 'Remote',
                            'description': job.get('description', ''),
                            'requirements': ', '.join(job.get('tags', [])),
                            'salary_range': f"${job.get('salary_min', 0)}-${job.get('salary_max', 0)}" if job.get('salary_min') else 'Not specified',
                            'job_type': 'Remote',
                            'experience_level': 'Mid-level',
                            'posted_date': datetime.now().strftime('%Y-%m-%d'),
                            'application_url': job.get('url', ''),
                            'source': 'RemoteOK'
                        })
                        
                return relevant_jobs
        except Exception as e:
            logger.error(f"Error searching RemoteOK: {e}")
            
        return []
        
    def _is_job_relevant(self, job: Dict, user_profile: UserProfile) -> bool:
        """Check if a job is relevant to the user profile"""
        job_text = f"{job.get('position', '')} {job.get('description', '')} {' '.join(job.get('tags', []))}"
        job_text_lower = job_text.lower()
        
        # Check if any user skills match job requirements
        skill_matches = sum(1 for skill in user_profile.skills if skill.lower() in job_text_lower)
        
        # Check if any user keywords match
        keyword_matches = sum(1 for keyword in user_profile.keywords[:10] if keyword in job_text_lower)
        
        return skill_matches >= 2 or keyword_matches >= 3
        
    def _generate_mock_jobs(self, user_profile: UserProfile, limit: int) -> List[Dict[str, Any]]:
        """Generate mock job listings for demonstration"""
        mock_companies = [
            'TechCorp', 'InnovateLabs', 'DataDriven Inc', 'CloudFirst Solutions',
            'AI Dynamics', 'NextGen Systems', 'Digital Pioneers', 'SmartTech',
            'FutureSoft', 'CodeCrafters', 'ByteBuilders', 'PixelPerfect'
        ]
        
        job_templates = [
            {
                'title': 'Software Engineer',
                'description': 'Develop and maintain software applications using modern technologies.',
                'requirements': 'Programming experience, problem-solving skills, team collaboration'
            },
            {
                'title': 'Data Analyst',
                'description': 'Analyze data to provide insights and support business decisions.',
                'requirements': 'Data analysis, SQL, Excel, statistical knowledge'
            },
            {
                'title': 'Product Manager',
                'description': 'Lead product development and strategy initiatives.',
                'requirements': 'Product management, strategic thinking, communication skills'
            },
            {
                'title': 'UX Designer',
                'description': 'Design user-friendly interfaces and improve user experience.',
                'requirements': 'UI/UX design, Figma, user research, prototyping'
            }
        ]
        
        mock_jobs = []
        for i in range(limit):
            template = random.choice(job_templates)
            company = random.choice(mock_companies)
            
            # Customize based on user profile
            if user_profile.skills:
                relevant_skills = random.sample(user_profile.skills, min(3, len(user_profile.skills)))
                template['requirements'] += f", {', '.join(relevant_skills)}"
            
            mock_jobs.append({
                'title': template['title'],
                'company': company,
                'location': user_profile.location or 'Remote',
                'description': template['description'],
                'requirements': template['requirements'],
                'salary_range': f"${random.randint(60, 150)}k - ${random.randint(80, 200)}k",
                'job_type': random.choice(['Full-time', 'Part-time', 'Contract']),
                'experience_level': random.choice(['Entry-level', 'Mid-level', 'Senior-level']),
                'posted_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'application_url': f'https://example.com/jobs/{i}',
                'source': 'Mock Data'
            })
            
        return mock_jobs
        
    def calculate_job_match_scores(self, jobs: List[Dict[str, Any]], user_profile: UserProfile) -> List[JobOpportunity]:
        """Calculate match scores for jobs based on user profile"""
        logger.info("Calculating job match scores")
        
        job_opportunities = []
        
        for job in jobs:
            match_score, matching_skills, missing_skills = self._calculate_single_match_score(job, user_profile)
            
            opportunity = JobOpportunity(
                title=job['title'],
                company=job['company'],
                location=job['location'],
                description=job['description'],
                requirements=job['requirements'],
                salary_range=job['salary_range'],
                job_type=job['job_type'],
                experience_level=job['experience_level'],
                posted_date=job['posted_date'],
                application_url=job['application_url'],
                match_score=match_score,
                matching_skills=matching_skills,
                missing_skills=missing_skills
            )
            
            job_opportunities.append(opportunity)
            
        # Sort by match score
        job_opportunities.sort(key=lambda x: x.match_score, reverse=True)
        
        return job_opportunities
        
    def _calculate_single_match_score(self, job: Dict[str, Any], user_profile: UserProfile) -> Tuple[float, List[str], List[str]]:
        """Calculate match score for a single job"""
        job_text = f"{job['title']} {job['description']} {job['requirements']}".lower()
        
        # Calculate skill matches
        matching_skills = []
        for skill in user_profile.skills:
            if skill.lower() in job_text:
                matching_skills.append(skill)
                
        # Calculate missing skills (skills mentioned in job but not in user profile)
        missing_skills = []
        for skill in self.all_skills:
            if skill.lower() in job_text and skill not in user_profile.skills:
                missing_skills.append(skill)
                
        # Calculate base match score
        skill_match_ratio = len(matching_skills) / max(len(user_profile.skills), 1)
        
        # Bonus for location match
        location_bonus = 0.1 if user_profile.location.lower() in job['location'].lower() else 0
        
        # Bonus for experience level match
        experience_bonus = 0.1 if self._experience_level_matches(user_profile.experience_years, job['experience_level']) else 0
        
        # Calculate final score (0-100)
        match_score = min(100, (skill_match_ratio * 70) + (location_bonus * 100) + (experience_bonus * 100))
        
        return match_score, matching_skills[:5], missing_skills[:5]
        
    def _experience_level_matches(self, years: int, level: str) -> bool:
        """Check if experience years match the job level"""
        level_lower = level.lower()
        if 'entry' in level_lower or 'junior' in level_lower:
            return years <= 2
        elif 'mid' in level_lower:
            return 2 <= years <= 5
        elif 'senior' in level_lower:
            return years >= 5
        return True
        
    def get_job_recommendations(self, resume_text: str, limit: int = 20) -> Dict[str, Any]:
        """Main method to get job recommendations for a resume"""
        try:
            # Extract user profile
            user_profile = self.extract_user_profile(resume_text)
            
            # Search for jobs
            jobs = self.search_jobs_online(user_profile, limit * 2)  # Get more to filter
            
            # Calculate match scores
            job_opportunities = self.calculate_job_match_scores(jobs, user_profile)
            
            # Filter and limit results
            top_jobs = job_opportunities[:limit]
            
            # Generate insights
            insights = self._generate_insights(user_profile, top_jobs)
            
            return {
                'success': True,
                'user_profile': {
                    'name': user_profile.name,
                    'location': user_profile.location,
                    'skills': user_profile.skills,
                    'experience_years': user_profile.experience_years,
                    'education_level': user_profile.education_level,
                    'job_titles': user_profile.job_titles
                },
                'job_recommendations': [
                    {
                        'title': job.title,
                        'company': job.company,
                        'location': job.location,
                        'description': job.description[:200] + '...' if len(job.description) > 200 else job.description,
                        'requirements': job.requirements,
                        'salary_range': job.salary_range,
                        'job_type': job.job_type,
                        'experience_level': job.experience_level,
                        'posted_date': job.posted_date,
                        'application_url': job.application_url,
                        'match_score': round(job.match_score, 1),
                        'matching_skills': job.matching_skills,
                        'missing_skills': job.missing_skills
                    }
                    for job in top_jobs
                ],
                'insights': insights,
                'total_jobs_found': len(jobs),
                'recommendations_count': len(top_jobs)
            }
            
        except Exception as e:
            logger.error(f"Error getting job recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_recommendations': [],
                'insights': {}
            }
            
    def _generate_insights(self, user_profile: UserProfile, jobs: List[JobOpportunity]) -> Dict[str, Any]:
        """Generate insights about job market and recommendations"""
        if not jobs:
            return {}
            
        # Calculate average match score
        avg_match_score = sum(job.match_score for job in jobs) / len(jobs)
        
        # Find most common skills required
        all_missing_skills = []
        for job in jobs:
            all_missing_skills.extend(job.missing_skills)
            
        from collections import Counter
        skill_gaps = Counter(all_missing_skills).most_common(5)
        
        # Find most common job types and locations
        job_types = Counter(job.job_type for job in jobs).most_common(3)
        locations = Counter(job.location for job in jobs).most_common(3)
        
        return {
            'average_match_score': round(avg_match_score, 1),
            'skill_gaps': [{'skill': skill, 'frequency': count} for skill, count in skill_gaps],
            'popular_job_types': [{'type': jtype, 'count': count} for jtype, count in job_types],
            'popular_locations': [{'location': loc, 'count': count} for loc, count in locations],
            'recommendations': [
                f"Consider learning {skill_gaps[0][0]} to improve your job prospects" if skill_gaps else "Your skills are well-aligned with available opportunities",
                f"Most opportunities are {job_types[0][0]} positions" if job_types else "Various job types available",
                f"Consider expanding your search to {locations[0][0]}" if locations and len(locations) > 1 else "Good location match"
            ]
        }

# Initialize the job matcher
job_matcher = IntelligentJobMatcher()