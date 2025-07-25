# PassResume - AI-Powered Resume Optimizer 🚀

**Latest Update**: Advanced AI Processing System with automatic GitHub deployments enabled!

## Features
- ✅ ATS Resume Analysis
- ✅ AI-Powered Improvements  
- ✅ Advanced AI Processing (NEW!)
- ✅ Admin Dashboard (NEW!)
- ✅ Automatic Deployments (NEW!)

## Live Demo
🌐 **Production**: https://passresume.netlify.app

---
*Automatically deploys from GitHub main branch*

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
# Repository is private - contact owner for access
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

1. Contact the development team directly
2. Check the documentation for troubleshooting
3. Reach out via email for technical support

## 🙏 Acknowledgments

- Built with modern web technologies
- Inspired by the need for better ATS optimization tools
- Community-driven development approach

---

**Made with ❤️ for job seekers everywhere**