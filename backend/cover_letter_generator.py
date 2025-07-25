"""
AI-Powered Cover Letter Generator for passResume
Generates tailored cover letters based on resume content and job descriptions
"""

import re
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import random

class CoverLetterGenerator:
    """AI-powered cover letter generation system"""
    
    def __init__(self, database_path: str = 'resume_analyzer.db'):
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # Cover letter templates
        self.templates = {
            'professional': {
                'name': 'Professional',
                'description': 'Formal, traditional cover letter format',
                'structure': [
                    'header',
                    'date',
                    'employer_address',
                    'salutation',
                    'opening_paragraph',
                    'body_paragraph_1',
                    'body_paragraph_2',
                    'closing_paragraph',
                    'sign_off'
                ]
            },
            'modern': {
                'name': 'Modern',
                'description': 'Contemporary, engaging cover letter format',
                'structure': [
                    'header',
                    'date',
                    'salutation',
                    'hook_opening',
                    'value_proposition',
                    'achievement_highlight',
                    'company_connection',
                    'call_to_action',
                    'sign_off'
                ]
            },
            'creative': {
                'name': 'Creative',
                'description': 'Dynamic, personality-driven cover letter',
                'structure': [
                    'header',
                    'date',
                    'personal_salutation',
                    'story_opening',
                    'skills_narrative',
                    'passion_statement',
                    'future_vision',
                    'enthusiastic_close',
                    'sign_off'
                ]
            }
        }
        
        # Industry-specific keywords and phrases
        self.industry_keywords = {
            'technology': ['innovative', 'scalable', 'agile', 'cutting-edge', 'digital transformation'],
            'healthcare': ['patient-centered', 'evidence-based', 'compassionate', 'clinical excellence'],
            'finance': ['analytical', 'risk management', 'strategic planning', 'financial modeling'],
            'marketing': ['brand awareness', 'customer engagement', 'data-driven', 'creative campaigns'],
            'education': ['student-centered', 'curriculum development', 'educational excellence'],
            'sales': ['relationship building', 'revenue growth', 'client acquisition', 'target achievement'],
            'general': ['results-driven', 'collaborative', 'detail-oriented', 'problem-solving']
        }
    
    def extract_resume_profile(self, resume_text: str) -> Dict[str, Any]:
        """Extract key information from resume for cover letter personalization"""
        profile = {
            'name': '',
            'email': '',
            'phone': '',
            'skills': [],
            'experience': [],
            'achievements': [],
            'education': [],
            'industry': 'general'
        }
        
        lines = resume_text.split('\n')
        
        # Extract name (usually first non-empty line)
        for line in lines:
            if line.strip() and not re.search(r'@|phone|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line.lower()):
                profile['name'] = line.strip()
                break
        
        # Extract contact information
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
        if email_match:
            profile['email'] = email_match.group()
        
        phone_match = re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
        if phone_match:
            profile['phone'] = phone_match.group()
        
        # Extract skills
        skills_section = re.search(r'(SKILLS|TECHNICAL SKILLS|COMPETENCIES)[\s\S]*?(?=\n[A-Z]|\n\n|$)', resume_text, re.IGNORECASE)
        if skills_section:
            skills_text = skills_section.group(0)
            # Extract skills from bullet points or comma-separated lists
            skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z\s\+\#]{2,})', skills_text)
            profile['skills'] = [skill.strip() for skill in skills if len(skill.strip()) > 2][:10]
        
        # Extract experience
        experience_section = re.search(r'(EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT)[\s\S]*?(?=\n[A-Z]|\n\n|$)', resume_text, re.IGNORECASE)
        if experience_section:
            exp_text = experience_section.group(0)
            # Extract job titles and companies
            job_patterns = [
                r'([A-Z][A-Za-z\s]+(?:Manager|Director|Analyst|Engineer|Developer|Specialist|Coordinator))',
                r'([A-Z][A-Za-z\s]+)\s+(?:at|@)\s+([A-Z][A-Za-z\s&]+)'
            ]
            for pattern in job_patterns:
                matches = re.findall(pattern, exp_text)
                profile['experience'].extend(matches[:3])
        
        # Extract achievements (bullet points with numbers/percentages)
        achievement_patterns = [
            r'[•\-\*]\s*([^•\-\*\n]*(?:\d+%|\$\d+|increased|improved|reduced|achieved)[^•\-\*\n]*)',
            r'[•\-\*]\s*([^•\-\*\n]*(?:led|managed|developed|implemented|created)[^•\-\*\n]*)'
        ]
        for pattern in achievement_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            profile['achievements'].extend([match.strip() for match in matches[:5]])
        
        # Determine industry based on skills and experience
        profile['industry'] = self._determine_industry(profile['skills'], profile['experience'])
        
        return profile
    
    def _determine_industry(self, skills: List[str], experience: List[Any]) -> str:
        """Determine industry based on skills and experience"""
        skills_text = ' '.join(skills).lower()
        exp_text = ' '.join([str(exp) for exp in experience]).lower()
        combined_text = skills_text + ' ' + exp_text
        
        industry_indicators = {
            'technology': ['python', 'java', 'javascript', 'software', 'developer', 'programming', 'api', 'database'],
            'healthcare': ['medical', 'patient', 'clinical', 'healthcare', 'nursing', 'doctor', 'hospital'],
            'finance': ['financial', 'accounting', 'investment', 'banking', 'analyst', 'finance', 'budget'],
            'marketing': ['marketing', 'social media', 'advertising', 'brand', 'campaign', 'seo', 'content'],
            'education': ['teaching', 'education', 'curriculum', 'student', 'academic', 'school', 'training'],
            'sales': ['sales', 'business development', 'client', 'revenue', 'account management', 'crm']
        }
        
        for industry, indicators in industry_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                return industry
        
        return 'general'
    
    def generate_cover_letter(self, resume_text: str, job_description: str = "", 
                            company_name: str = "", position_title: str = "",
                            template: str = 'professional') -> Dict[str, Any]:
        """Generate a tailored cover letter"""
        try:
            # Extract resume profile
            profile = self.extract_resume_profile(resume_text)
            
            # Analyze job description if provided
            job_analysis = self._analyze_job_description(job_description) if job_description else {}
            
            # Generate cover letter content
            cover_letter_content = self._generate_content(
                profile, job_analysis, company_name, position_title, template
            )
            
            # Track generation
            self._track_cover_letter_generation(profile.get('name', 'Unknown'), template)
            
            return {
                'status': 'success',
                'cover_letter': cover_letter_content,
                'template_used': template,
                'personalization_score': self._calculate_personalization_score(profile, job_analysis),
                'suggestions': self._generate_improvement_suggestions(profile, job_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Cover letter generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'fallback_letter': self._generate_fallback_letter(resume_text, company_name, position_title)
            }
    
    def _analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description to extract key requirements"""
        analysis = {
            'required_skills': [],
            'preferred_skills': [],
            'responsibilities': [],
            'company_values': [],
            'experience_level': '',
            'industry_keywords': []
        }
        
        # Extract required skills
        required_section = re.search(r'(required|must have|essential)[\s\S]*?(?=preferred|nice to have|$)', 
                                   job_description, re.IGNORECASE)
        if required_section:
            skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z\s\+\#]{2,})', required_section.group(0))
            analysis['required_skills'] = [skill.strip() for skill in skills[:8]]
        
        # Extract preferred skills
        preferred_section = re.search(r'(preferred|nice to have|bonus)[\s\S]*?(?=responsibilities|$)', 
                                    job_description, re.IGNORECASE)
        if preferred_section:
            skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z\s\+\#]{2,})', preferred_section.group(0))
            analysis['preferred_skills'] = [skill.strip() for skill in skills[:5]]
        
        # Extract responsibilities
        resp_section = re.search(r'(responsibilities|duties|role)[\s\S]*?(?=requirements|qualifications|$)', 
                               job_description, re.IGNORECASE)
        if resp_section:
            responsibilities = re.findall(r'[•\-\*]\s*([^•\-\*\n]+)', resp_section.group(0))
            analysis['responsibilities'] = [resp.strip() for resp in responsibilities[:5]]
        
        # Determine experience level
        if re.search(r'senior|lead|principal|architect', job_description, re.IGNORECASE):
            analysis['experience_level'] = 'senior'
        elif re.search(r'junior|entry|associate|graduate', job_description, re.IGNORECASE):
            analysis['experience_level'] = 'junior'
        else:
            analysis['experience_level'] = 'mid'
        
        return analysis
    
    def _generate_content(self, profile: Dict, job_analysis: Dict, 
                         company_name: str, position_title: str, template: str) -> str:
        """Generate the actual cover letter content"""
        template_config = self.templates.get(template, self.templates['professional'])
        industry_keywords = self.industry_keywords.get(profile['industry'], self.industry_keywords['general'])
        
        # Header
        header = f"{profile['name']}\n"
        if profile['email']:
            header += f"{profile['email']}"
        if profile['phone']:
            header += f" | {profile['phone']}"
        header += "\n\n"
        
        # Date
        date = datetime.now().strftime("%B %d, %Y") + "\n\n"
        
        # Salutation
        if company_name:
            salutation = f"Dear {company_name} Hiring Manager,\n\n"
        else:
            salutation = "Dear Hiring Manager,\n\n"
        
        # Opening paragraph
        opening = self._generate_opening_paragraph(profile, position_title, company_name, template)
        
        # Body paragraphs
        body1 = self._generate_experience_paragraph(profile, job_analysis, industry_keywords)
        body2 = self._generate_skills_paragraph(profile, job_analysis, industry_keywords)
        
        # Closing paragraph
        closing = self._generate_closing_paragraph(company_name, template)
        
        # Sign off
        sign_off = f"Sincerely,\n{profile['name']}"
        
        # Combine all parts
        cover_letter = header + date + salutation + opening + "\n\n" + body1 + "\n\n" + body2 + "\n\n" + closing + "\n\n" + sign_off
        
        return cover_letter
    
    def _generate_opening_paragraph(self, profile: Dict, position_title: str, 
                                  company_name: str, template: str) -> str:
        """Generate opening paragraph based on template style"""
        name = profile['name'] or "I"
        
        if template == 'modern':
            openings = [
                f"As a {random.choice(['passionate', 'dedicated', 'results-driven'])} professional with expertise in {', '.join(profile['skills'][:2])}, I was excited to discover the {position_title} opportunity at {company_name}.",
                f"Your {position_title} position immediately caught my attention because it perfectly aligns with my background in {', '.join(profile['skills'][:2])} and my passion for {random.choice(['innovation', 'excellence', 'growth'])}.",
                f"I am writing to express my strong interest in the {position_title} role at {company_name}, where I can leverage my {len(profile['experience'])}+ years of experience to drive {random.choice(['success', 'innovation', 'growth'])}."
            ]
        elif template == 'creative':
            openings = [
                f"Imagine a professional who combines {', '.join(profile['skills'][:2])} with {random.choice(['creativity', 'innovation', 'strategic thinking'])} – that's exactly what I bring to the {position_title} role at {company_name}.",
                f"When I saw your {position_title} posting, I knew this was the opportunity I've been preparing for throughout my career in {profile['industry']}.",
                f"Three things excite me about the {position_title} role at {company_name}: the opportunity to {random.choice(['innovate', 'lead', 'create'])}, the chance to work with {random.choice(['cutting-edge technology', 'talented teams', 'industry leaders'])}, and the potential to {random.choice(['make an impact', 'drive growth', 'solve challenges'])}."
            ]
        else:  # professional
            openings = [
                f"I am writing to express my interest in the {position_title} position at {company_name}. With my background in {', '.join(profile['skills'][:2])}, I am confident I would be a valuable addition to your team.",
                f"I am pleased to submit my application for the {position_title} role at {company_name}. My experience in {profile['industry']} and proven track record make me an ideal candidate for this position.",
                f"I would like to be considered for the {position_title} position at {company_name}. My {len(profile['experience'])}+ years of experience and expertise in {', '.join(profile['skills'][:2])} align well with your requirements."
            ]
        
        return random.choice(openings)
    
    def _generate_experience_paragraph(self, profile: Dict, job_analysis: Dict, 
                                     industry_keywords: List[str]) -> str:
        """Generate paragraph highlighting relevant experience"""
        achievements = profile['achievements'][:2] if profile['achievements'] else []
        skills = profile['skills'][:3] if profile['skills'] else []
        
        if achievements:
            achievement_text = f"In my previous roles, I have {achievements[0].lower()}"
            if len(achievements) > 1:
                achievement_text += f" and {achievements[1].lower()}"
        else:
            achievement_text = f"Throughout my career, I have consistently delivered {random.choice(['exceptional results', 'innovative solutions', 'strategic value'])}"
        
        skills_text = f"My expertise in {', '.join(skills)} has enabled me to {random.choice(['drive success', 'exceed expectations', 'deliver results'])}"
        
        keyword = random.choice(industry_keywords)
        
        return f"{achievement_text}. {skills_text}, making me well-positioned to contribute to your team's {keyword} initiatives."
    
    def _generate_skills_paragraph(self, profile: Dict, job_analysis: Dict, 
                                 industry_keywords: List[str]) -> str:
        """Generate paragraph highlighting relevant skills"""
        skills = profile['skills'][:4] if profile['skills'] else ['problem-solving', 'communication', 'teamwork']
        required_skills = job_analysis.get('required_skills', [])[:3]
        
        # Match skills with job requirements
        matching_skills = [skill for skill in skills if any(req.lower() in skill.lower() for req in required_skills)]
        if not matching_skills:
            matching_skills = skills[:3]
        
        skills_text = f"My technical proficiency includes {', '.join(matching_skills)}"
        
        if required_skills:
            alignment_text = f"which directly aligns with your requirements for {', '.join(required_skills[:2])}"
        else:
            alignment_text = f"which positions me to excel in this {random.choice(['dynamic', 'challenging', 'innovative'])} role"
        
        keyword = random.choice(industry_keywords)
        
        return f"{skills_text}, {alignment_text}. I am particularly excited about the opportunity to apply my {keyword} approach to help your organization achieve its goals."
    
    def _generate_closing_paragraph(self, company_name: str, template: str) -> str:
        """Generate closing paragraph based on template style"""
        if template == 'modern':
            closings = [
                f"I would welcome the opportunity to discuss how my background and enthusiasm can contribute to {company_name}'s continued success. Thank you for your consideration.",
                f"I am excited about the possibility of joining {company_name} and contributing to your team's innovative work. I look forward to hearing from you soon.",
                f"Thank you for considering my application. I am eager to discuss how I can help {company_name} achieve its strategic objectives."
            ]
        elif template == 'creative':
            closings = [
                f"I am genuinely excited about the opportunity to bring my passion and expertise to {company_name}. Let's connect to discuss how we can create something amazing together.",
                f"The prospect of contributing to {company_name}'s mission energizes me. I would love to discuss how my unique perspective can add value to your team.",
                f"I believe great things happen when passion meets opportunity, and I see both in this role at {company_name}. I look forward to our conversation."
            ]
        else:  # professional
            closings = [
                f"I would appreciate the opportunity to discuss my qualifications further. Thank you for your time and consideration.",
                f"I look forward to the possibility of contributing to {company_name}'s success. Thank you for reviewing my application.",
                f"I am confident that my skills and experience make me a strong candidate for this position. I look forward to hearing from you."
            ]
        
        return random.choice(closings)
    
    def _generate_fallback_letter(self, resume_text: str, company_name: str, position_title: str) -> str:
        """Generate a basic cover letter when full generation fails"""
        profile = self.extract_resume_profile(resume_text)
        
        return f"""Dear Hiring Manager,

I am writing to express my interest in the {position_title} position at {company_name}. With my background and experience, I believe I would be a valuable addition to your team.

My skills include {', '.join(profile['skills'][:3]) if profile['skills'] else 'problem-solving, communication, and teamwork'}, which I have developed through my professional experience. I am passionate about contributing to organizational success and am excited about the opportunity to bring my expertise to your company.

I would welcome the opportunity to discuss my qualifications further. Thank you for your consideration.

Sincerely,
{profile['name'] or 'Applicant'}"""
    
    def _calculate_personalization_score(self, profile: Dict, job_analysis: Dict) -> int:
        """Calculate how well the cover letter is personalized"""
        score = 50  # Base score
        
        # Add points for profile completeness
        if profile['name']: score += 10
        if profile['skills']: score += 10
        if profile['achievements']: score += 10
        if profile['experience']: score += 10
        
        # Add points for job analysis
        if job_analysis.get('required_skills'): score += 5
        if job_analysis.get('responsibilities'): score += 5
        
        return min(100, score)
    
    def _generate_improvement_suggestions(self, profile: Dict, job_analysis: Dict) -> List[str]:
        """Generate suggestions for improving the cover letter"""
        suggestions = []
        
        if not profile['achievements']:
            suggestions.append("Add specific achievements with quantifiable results to strengthen your cover letter")
        
        if len(profile['skills']) < 3:
            suggestions.append("Include more relevant skills from your resume to better match job requirements")
        
        if not job_analysis:
            suggestions.append("Provide a job description to create a more targeted cover letter")
        
        if not profile['experience']:
            suggestions.append("Highlight specific work experience to demonstrate your qualifications")
        
        return suggestions
    
    def _track_cover_letter_generation(self, user_name: str, template: str):
        """Track cover letter generation for analytics"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, activity_data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                0,  # Anonymous user
                'cover_letter_generated',
                json.dumps({'template': template, 'user_name': user_name}),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.warning(f"Failed to track cover letter generation: {e}")
    
    def get_available_templates(self) -> List[Dict]:
        """Get list of available cover letter templates"""
        return [
            {
                'name': name,
                'display_name': config['name'],
                'description': config['description']
            }
            for name, config in self.templates.items()
        ]
    
    def get_generation_statistics(self) -> Dict:
        """Get cover letter generation statistics"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as total_generated
                FROM user_activity
                WHERE activity_type = 'cover_letter_generated'
            ''')
            
            total = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT activity_data, COUNT(*) as count
                FROM user_activity
                WHERE activity_type = 'cover_letter_generated'
                GROUP BY activity_data
            ''')
            
            template_stats = {}
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[0])
                    template = data.get('template', 'unknown')
                    template_stats[template] = row[1]
                except:
                    continue
            
            conn.close()
            
            return {
                'total_generated': total,
                'template_usage': template_stats,
                'most_popular_template': max(template_stats.items(), key=lambda x: x[1])[0] if template_stats else 'professional'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get generation statistics: {e}")
            return {'total_generated': 0, 'template_usage': {}, 'most_popular_template': 'professional'}