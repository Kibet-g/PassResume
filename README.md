# 🚀 Open Resume Auditor

A free, open-source AI-powered resume auditing tool that ensures ATS compliance and increases your chances of landing interviews.

## 🔎 Vision Statement
Empower job seekers with a free, AI-powered resume auditing tool that ensures compliance with ATS standards and increases their chances of landing interviews.

## 🌐 Tech Stack
- **Frontend**: Next.js (TypeScript), Tailwind CSS
- **Backend**: Flask (Python)
- **Database**: SQLite
- **AI/NLP**: spaCy, sentence-transformers
- **File Processing**: pdfplumber, python-docx

## 🔄 Features
1. **Resume Upload & Parsing** - Drag-and-drop file upload (PDF/DOCX)
2. **ATS Compliance Audit** - Comprehensive ATS score (0-100%)
3. **Job Description Matching** - Keyword analysis and matching
4. **AI-Powered Suggestions** - Formatting and content improvements

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 📊 API Endpoints
- `POST /api/upload` - Upload and parse resume
- `POST /api/analyze` - Analyze resume against job description
- `GET /api/resume/{id}` - Get resume analysis results

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Built with ❤️ for the job-seeking community
- Powered by open-source AI/ML libraries