"""
Advanced PDF Export System for Enhanced Resume Generation
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, darkblue, gray
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import logging

logger = logging.getLogger(__name__)

class AdvancedPDFExporter:
    """Advanced PDF export system with multiple templates and ATS optimization"""
    
    def __init__(self):
        self.templates = {
            'professional': {
                'name': 'Professional',
                'description': 'Clean, traditional layout optimized for ATS',
                'colors': {
                    'primary': darkblue,
                    'secondary': gray,
                    'text': black
                },
                'fonts': {
                    'header': 'Helvetica-Bold',
                    'subheader': 'Helvetica-Bold',
                    'body': 'Helvetica',
                    'emphasis': 'Helvetica-Oblique'
                },
                'spacing': {
                    'section': 18,
                    'subsection': 12,
                    'paragraph': 6
                }
            },
            'modern': {
                'name': 'Modern',
                'description': 'Contemporary design with subtle styling',
                'colors': {
                    'primary': colors.HexColor('#2C3E50'),
                    'secondary': colors.HexColor('#34495E'),
                    'text': black
                },
                'fonts': {
                    'header': 'Helvetica-Bold',
                    'subheader': 'Helvetica-Bold',
                    'body': 'Helvetica',
                    'emphasis': 'Helvetica-Oblique'
                },
                'spacing': {
                    'section': 16,
                    'subsection': 10,
                    'paragraph': 5
                }
            },
            'executive': {
                'name': 'Executive',
                'description': 'Sophisticated layout for senior positions',
                'colors': {
                    'primary': colors.HexColor('#1A1A1A'),
                    'secondary': colors.HexColor('#4A4A4A'),
                    'text': black
                },
                'fonts': {
                    'header': 'Times-Bold',
                    'subheader': 'Times-Bold',
                    'body': 'Times-Roman',
                    'emphasis': 'Times-Italic'
                },
                'spacing': {
                    'section': 20,
                    'subsection': 14,
                    'paragraph': 7
                }
            }
        }
        
        self.export_folder = 'exports'
        os.makedirs(self.export_folder, exist_ok=True)
    
    def parse_resume_sections(self, text: str) -> Dict[str, str]:
        """Parse resume text into structured sections"""
        sections = {}
        current_section = 'header'
        current_content = []
        
        lines = text.split('\n')
        
        section_patterns = {
            'contact': r'(?i)^(contact|personal\s+information)',
            'summary': r'(?i)^(summary|profile|objective|about)',
            'experience': r'(?i)^(experience|work\s+history|employment|professional\s+experience)',
            'education': r'(?i)^(education|academic|qualifications)',
            'skills': r'(?i)^(skills|technical\s+skills|competencies)',
            'projects': r'(?i)^(projects|portfolio)',
            'certifications': r'(?i)^(certifications|certificates|licenses)',
            'awards': r'(?i)^(awards|honors|achievements)',
            'references': r'(?i)^(references)'
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def create_pdf_styles(self, template_name: str) -> Dict:
        """Create PDF styles based on template"""
        template = self.templates.get(template_name, self.templates['professional'])
        styles = getSampleStyleSheet()
        
        custom_styles = {
            'name': ParagraphStyle(
                'CustomName',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=template['colors']['primary'],
                fontName=template['fonts']['header'],
                alignment=TA_CENTER,
                spaceAfter=12
            ),
            'contact': ParagraphStyle(
                'CustomContact',
                parent=styles['Normal'],
                fontSize=10,
                textColor=template['colors']['text'],
                fontName=template['fonts']['body'],
                alignment=TA_CENTER,
                spaceAfter=18
            ),
            'section_header': ParagraphStyle(
                'CustomSectionHeader',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=template['colors']['primary'],
                fontName=template['fonts']['subheader'],
                borderWidth=1,
                borderColor=template['colors']['primary'],
                borderPadding=3,
                spaceBefore=template['spacing']['section'],
                spaceAfter=template['spacing']['subsection']
            ),
            'job_title': ParagraphStyle(
                'CustomJobTitle',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=template['colors']['secondary'],
                fontName=template['fonts']['subheader'],
                spaceBefore=8,
                spaceAfter=4
            ),
            'company': ParagraphStyle(
                'CustomCompany',
                parent=styles['Normal'],
                fontSize=11,
                textColor=template['colors']['text'],
                fontName=template['fonts']['emphasis'],
                spaceAfter=4
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                textColor=template['colors']['text'],
                fontName=template['fonts']['body'],
                alignment=TA_JUSTIFY,
                spaceAfter=template['spacing']['paragraph']
            ),
            'bullet': ParagraphStyle(
                'CustomBullet',
                parent=styles['Normal'],
                fontSize=10,
                textColor=template['colors']['text'],
                fontName=template['fonts']['body'],
                leftIndent=20,
                bulletIndent=10,
                spaceAfter=3
            )
        }
        
        return custom_styles
    
    def export_resume_to_pdf(self, 
                           refined_text: str, 
                           template_name: str = 'professional',
                           filename: Optional[str] = None) -> Dict:
        """Export refined resume to PDF with specified template"""
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"resume_{template_name}_{timestamp}.pdf"
            
            filepath = os.path.join(self.export_folder, filename)
            
            # Parse resume into sections
            sections = self.parse_resume_sections(refined_text)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get styles
            styles = self.create_pdf_styles(template_name)
            
            # Build PDF content
            story = []
            
            # Add header section (name and contact)
            if 'header' in sections:
                story.extend(self._build_header_section(sections['header'], styles))
            
            # Add summary/objective
            if 'summary' in sections:
                story.extend(self._build_section('PROFESSIONAL SUMMARY', sections['summary'], styles))
            
            # Add experience
            if 'experience' in sections:
                story.extend(self._build_experience_section(sections['experience'], styles))
            
            # Add education
            if 'education' in sections:
                story.extend(self._build_section('EDUCATION', sections['education'], styles))
            
            # Add skills
            if 'skills' in sections:
                story.extend(self._build_skills_section(sections['skills'], styles))
            
            # Add projects
            if 'projects' in sections:
                story.extend(self._build_section('PROJECTS', sections['projects'], styles))
            
            # Add certifications
            if 'certifications' in sections:
                story.extend(self._build_section('CERTIFICATIONS', sections['certifications'], styles))
            
            # Add awards
            if 'awards' in sections:
                story.extend(self._build_section('AWARDS & HONORS', sections['awards'], styles))
            
            # Build PDF
            doc.build(story)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_kb = round(file_size / 1024, 1)
            
            return {
                'success': True,
                'filepath': filepath,
                'filename': filename,
                'template_used': template_name,
                'file_size_kb': file_size_kb,
                'sections_included': list(sections.keys()),
                'ats_optimized': True
            }
            
        except Exception as e:
            logger.error(f"Error exporting resume to PDF: {e}")
            return {
                'success': False,
                'error': str(e),
                'filepath': None
            }
    
    def _build_header_section(self, header_text: str, styles: Dict) -> List:
        """Build header section with name and contact info"""
        elements = []
        lines = [line.strip() for line in header_text.split('\n') if line.strip()]
        
        if lines:
            # First line is usually the name
            name = lines[0]
            elements.append(Paragraph(name, styles['name']))
            
            # Remaining lines are contact info
            if len(lines) > 1:
                contact_info = ' | '.join(lines[1:])
                elements.append(Paragraph(contact_info, styles['contact']))
        
        return elements
    
    def _build_section(self, title: str, content: str, styles: Dict) -> List:
        """Build a standard section with title and content"""
        elements = []
        
        # Section header
        elements.append(Paragraph(title, styles['section_header']))
        
        # Section content
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            if paragraph.startswith('•') or paragraph.startswith('-') or paragraph.startswith('*'):
                # Bullet point
                clean_text = re.sub(r'^[•\-\*]\s*', '', paragraph)
                elements.append(Paragraph(f"• {clean_text}", styles['bullet']))
            else:
                elements.append(Paragraph(paragraph, styles['body']))
        
        return elements
    
    def _build_experience_section(self, experience_text: str, styles: Dict) -> List:
        """Build experience section with structured job entries"""
        elements = []
        
        # Section header
        elements.append(Paragraph('PROFESSIONAL EXPERIENCE', styles['section_header']))
        
        # Parse job entries
        job_entries = self._parse_job_entries(experience_text)
        
        for job in job_entries:
            # Job title and company
            if job.get('title') and job.get('company'):
                elements.append(Paragraph(job['title'], styles['job_title']))
                company_line = job['company']
                if job.get('dates'):
                    company_line += f" | {job['dates']}"
                elements.append(Paragraph(company_line, styles['company']))
            
            # Job description/achievements
            if job.get('description'):
                for bullet in job['description']:
                    elements.append(Paragraph(f"• {bullet}", styles['bullet']))
            
            # Add space between jobs
            elements.append(Spacer(1, 8))
        
        return elements
    
    def _parse_job_entries(self, experience_text: str) -> List[Dict]:
        """Parse experience text into structured job entries"""
        jobs = []
        lines = [line.strip() for line in experience_text.split('\n') if line.strip()]
        
        current_job = {}
        current_bullets = []
        
        for line in lines:
            # Check if line looks like a job title or company
            if (not line.startswith('•') and 
                not line.startswith('-') and 
                not line.startswith('*') and
                len(line) > 10):
                
                # Save previous job if exists
                if current_job:
                    current_job['description'] = current_bullets
                    jobs.append(current_job)
                
                # Start new job
                current_job = {}
                current_bullets = []
                
                # Try to parse title, company, and dates
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    current_job['title'] = parts[0]
                    if len(parts) > 1:
                        current_job['company'] = parts[1]
                    if len(parts) > 2:
                        current_job['dates'] = parts[2]
                else:
                    current_job['title'] = line
            
            elif line.startswith(('•', '-', '*')):
                # Bullet point
                clean_bullet = re.sub(r'^[•\-\*]\s*', '', line)
                current_bullets.append(clean_bullet)
        
        # Save last job
        if current_job:
            current_job['description'] = current_bullets
            jobs.append(current_job)
        
        return jobs
    
    def _build_skills_section(self, skills_text: str, styles: Dict) -> List:
        """Build skills section with organized categories"""
        elements = []
        
        # Section header
        elements.append(Paragraph('TECHNICAL SKILLS', styles['section_header']))
        
        # Parse skills into categories if possible
        skills_data = self._parse_skills(skills_text)
        
        if isinstance(skills_data, dict):
            # Organized by categories
            for category, skills_list in skills_data.items():
                category_text = f"<b>{category}:</b> {', '.join(skills_list)}"
                elements.append(Paragraph(category_text, styles['body']))
        else:
            # Simple list
            elements.append(Paragraph(skills_text, styles['body']))
        
        return elements
    
    def _parse_skills(self, skills_text: str) -> Dict:
        """Parse skills text into categories"""
        skills_dict = {}
        lines = [line.strip() for line in skills_text.split('\n') if line.strip()]
        
        current_category = 'General'
        
        for line in lines:
            if ':' in line and not line.startswith(('•', '-', '*')):
                # Category line
                parts = line.split(':', 1)
                current_category = parts[0].strip()
                if len(parts) > 1:
                    skills_list = [s.strip() for s in parts[1].split(',') if s.strip()]
                    skills_dict[current_category] = skills_list
            else:
                # Skills line
                if current_category not in skills_dict:
                    skills_dict[current_category] = []
                
                # Remove bullet points and split by commas
                clean_line = re.sub(r'^[•\-\*]\s*', '', line)
                skills = [s.strip() for s in clean_line.split(',') if s.strip()]
                skills_dict[current_category].extend(skills)
        
        return skills_dict if skills_dict else skills_text
    
    def get_available_templates(self) -> List[Dict]:
        """Get list of available PDF templates"""
        return [
            {
                'name': name,
                'display_name': config['name'],
                'description': config['description']
            }
            for name, config in self.templates.items()
        ]
    
    def optimize_for_ats(self, text: str) -> str:
        """Optimize text for ATS compatibility"""
        # Remove special characters that might confuse ATS
        optimized = re.sub(r'[^\w\s\-\.\,\(\)\|\:\;\!\?]', '', text)
        
        # Ensure proper spacing
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # Remove excessive line breaks
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        return optimized.strip()

class ResumeAnalyticsTracker:
    """Track analytics for PDF exports and resume improvements"""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
    
    def track_pdf_export(self, user_id: int, template_used: str, file_size: float) -> bool:
        """Track PDF export event"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, activity_data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                'pdf_export',
                json.dumps({
                    'template': template_used,
                    'file_size_kb': file_size
                }),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error tracking PDF export: {e}")
            return False
    
    def get_export_statistics(self) -> Dict:
        """Get PDF export statistics"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get export counts by template
            cursor.execute('''
                SELECT activity_data, COUNT(*) as count
                FROM user_activity
                WHERE activity_type = 'pdf_export'
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
            
            # Get total exports
            cursor.execute('''
                SELECT COUNT(*) FROM user_activity WHERE activity_type = 'pdf_export'
            ''')
            total_exports = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_exports': total_exports,
                'template_usage': template_stats,
                'most_popular_template': max(template_stats.items(), key=lambda x: x[1])[0] if template_stats else None
            }
            
        except Exception as e:
            logger.error(f"Error getting export statistics: {e}")
            return {'total_exports': 0, 'template_usage': {}, 'most_popular_template': None}