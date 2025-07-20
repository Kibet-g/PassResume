# Open Resume Auditor 📄✨

An AI-powered resume auditing tool that helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) and improve their chances of landing interviews.

![Open Resume Auditor](https://img.shields.io/badge/Status-MVP-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Next.js](https://img.shields.io/badge/Next.js-14-black)

## 🚀 Features

- **ATS Compliance Scoring**: Get a comprehensive score based on resume structure and formatting
- **Keyword Analysis**: Compare your resume against job descriptions to identify missing keywords
- **AI-Powered Suggestions**: Receive intelligent recommendations for resume improvement
- **File Support**: Upload PDF and DOCX resume formats
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS
- **Real-time Analysis**: Instant feedback on resume quality

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **React Dropzone** - File upload functionality
- **Lucide React** - Modern icon library

### Backend
- **Flask** - Python web framework
- **SQLite** - Lightweight database
- **pdfplumber** - PDF text extraction
- **python-docx** - DOCX file processing
- **spaCy** - Natural language processing
- **sentence-transformers** - AI text analysis

## 📦 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 18 or higher
- npm or yarn

### Automated Setup

**Windows:**
```bash
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/open-resume-auditor.git
cd open-resume-auditor
```

2. **Backend Setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Frontend Setup:**
```bash
cd frontend
npm install
```

4. **Start the Application:**

Backend (Terminal 1):
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app_simple.py
```

Frontend (Terminal 2):
```bash
cd frontend
npm run dev
```

5. **Access the Application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## 📖 Usage

1. **Upload Resume**: Drag and drop or select your PDF/DOCX resume file
2. **Add Job Description** (Optional): Paste the target job description for keyword analysis
3. **Get Analysis**: View your ATS score, keyword matches, and improvement suggestions
4. **Implement Changes**: Follow the AI-powered recommendations to enhance your resume

## 🏗️ Project Structure

```
open-resume-auditor/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── app_simple.py       # Simplified Flask app for demo
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── frontend/               # Next.js application
│   ├── app/               # App router pages and components
│   ├── components/        # React components
│   ├── package.json       # Node.js dependencies
│   └── README.md         # Frontend documentation
├── setup.sh              # Unix setup script
├── setup.bat             # Windows setup script
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs
- Suggesting features
- Code contributions
- Development setup
- Code style guidelines

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Roadmap

- [ ] Advanced AI analysis with GPT integration
- [ ] Multiple resume format support
- [ ] Industry-specific optimization
- [ ] Resume template suggestions
- [ ] Batch processing capabilities
- [ ] Integration with job boards
- [ ] Mobile application

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/open-resume-auditor/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

## 🙏 Acknowledgments

- Built with modern web technologies
- Inspired by the need for better ATS optimization tools
- Community-driven development approach

---

**Made with ❤️ for job seekers everywhere**