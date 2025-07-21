# Contributing to Open Resume Auditor

Thank you for your interest in contributing to Open Resume Auditor! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

# Contributing to Open Resume Auditor

Thank you for your interest in contributing to Open Resume Auditor! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
- Contact the development team to report bugs
- Include detailed steps to reproduce the issue
- Provide information about your environment (OS, Python/Node.js versions)

### Suggesting Features
- Reach out to the team with feature requests
- Describe the feature and its benefits
- Include mockups or examples if applicable

### Code Contributions

1. **Contact the team for access**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test your changes**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add: feature description"
   ```
6. **Submit your changes for review**

## 🏗️ Development Setup

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## 📝 Code Style

### Python (Backend)
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

### TypeScript/React (Frontend)
- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Keep components small and focused

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm run test
```

## 📚 Documentation

- Update README.md for significant changes
- Add inline comments for complex logic
- Update API documentation for new endpoints

## 🎯 Priority Areas

We're especially looking for contributions in:

1. **AI/ML Improvements**
   - Better keyword extraction algorithms
   - Enhanced ATS compliance scoring
   - Custom LLM integration

2. **UI/UX Enhancements**
   - Mobile responsiveness improvements
   - Accessibility features
   - User experience optimizations

3. **Features**
   - Resume templates
   - Export functionality
   - Batch processing
   - Integration with job boards

4. **Performance**
   - Backend optimization
   - Frontend performance improvements
   - Database query optimization

## 🚀 Release Process

1. Features are developed in feature branches
2. Pull requests are reviewed by maintainers
3. Approved changes are merged to main
4. Releases are tagged and deployed

## 📞 Getting Help

- Contact the development team directly
- Email for questions and support
- Check existing documentation first

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Invited to join the core team for significant contributions

Thank you for helping make Open Resume Auditor better for everyone! 🎉