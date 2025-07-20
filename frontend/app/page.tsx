'use client'

import { useState, useEffect } from 'react'
import { Upload, FileText, Target, Zap, Github, Star, Brain, Shield, Download, TrendingUp, Database, CheckCircle, User, LogIn } from 'lucide-react'
import FileUpload from './components/FileUpload'
import AuthModal from './components/AuthModal'
import UserDashboard from './components/UserDashboard'
import { useAuth } from './contexts/AuthContext'
import axios from 'axios'

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1)
  const [resumeData, setResumeData] = useState<any>(null)
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [appStats, setAppStats] = useState<any>(null)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [showDashboard, setShowDashboard] = useState(false)
  const [authModalMode, setAuthModalMode] = useState<'login' | 'signup'>('login')

  const { user, isAuthenticated, loading } = useAuth()

  // Fetch app statistics on load
  useEffect(() => {
    fetchAppStats()
  }, [])

  const fetchAppStats = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/stats')
      if (response.ok) {
        const stats = await response.json()
        setAppStats(stats)
      }
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  const handleResumeUpload = (data: any) => {
    setResumeData(data)
    setCurrentStep(2)
    fetchAppStats()
  }

  const handleAnalyzeResume = async () => {
    if (!resumeData) return
    
    setIsAnalyzing(true)
    try {
      // Get the current resume data from the database
      const response = await axios.get(`http://127.0.0.1:5000/api/resume/${resumeData.resume_id}`)
      const resumeDetails = response.data
      
      setAnalysisResults({
        ats_score: resumeDetails.ats_score,
        ats_friendly: resumeDetails.ats_score >= 70,
        resume_id: resumeData.resume_id,
        auto_improved: resumeDetails.auto_improved || false,
        original_score: resumeDetails.original_ats_score || resumeDetails.ats_score
      })
      setCurrentStep(3)
    } catch (error) {
      console.error('Error analyzing resume:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleGenerateUpdatedResume = async () => {
    if (!resumeData) return
    
    setIsGenerating(true)
    try {
      // Call the auto-improve endpoint
      const response = await axios.post('http://127.0.0.1:5000/api/auto-improve-resume', {
        resume_id: resumeData.resume_id
      })
      
      const improvedData = response.data
      
      // Create and download the improved resume
      const blob = new Blob([improvedData.improved_text], { type: 'text/plain' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      
      // Determine file extension based on original upload
      const originalFileName = resumeData.file_name || 'resume'
      const fileExtension = originalFileName.includes('.') 
        ? originalFileName.split('.').pop() 
        : 'txt'
      
      link.download = `improved_resume_ats_${improvedData.final_ats_score}.${fileExtension}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
      
      // Update analysis results with new data
      setAnalysisResults(prev => ({
        ...prev,
        ats_score: improvedData.final_ats_score,
        improvement_applied: true,
        score_improvement: improvedData.final_ats_score - (prev?.original_score || prev?.ats_score || 0)
      }))
      
    } catch (error) {
      console.error('Error generating updated resume:', error)
      alert('Error generating updated resume. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  const resetProcess = () => {
    setCurrentStep(1)
    setResumeData(null)
    setAnalysisResults(null)
  }

  const handleAuthClick = (mode: 'login' | 'signup') => {
    setAuthModalMode(mode)
    setShowAuthModal(true)
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-primary-600 p-2 rounded-lg">
                <FileText className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">ATS Resume Optimizer</h1>
                <p className="text-sm text-gray-600">Upload → Analyze → Download Improved Resume</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {appStats && (
                <div className="flex items-center space-x-2 text-sm">
                  <div className="flex items-center space-x-1">
                    <Brain className="h-4 w-4 text-blue-600" />
                    <span className="text-gray-600">{appStats.model_status}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Database className="h-4 w-4 text-green-600" />
                    <span className="text-gray-600">{appStats.total_resumes} resumes</span>
                  </div>
                </div>
              )}
              
              {/* Authentication Section */}
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
              ) : isAuthenticated ? (
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setShowDashboard(true)}
                    className="flex items-center space-x-2 bg-primary-600 text-white px-3 py-2 rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    <User className="h-4 w-4" />
                    <span className="hidden sm:inline">Dashboard</span>
                  </button>
                  <div className="text-sm text-gray-600">
                    Welcome, {user?.full_name?.split(' ')[0]}
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleAuthClick('login')}
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors px-3 py-2"
                  >
                    <LogIn className="h-4 w-4" />
                    <span>Sign In</span>
                  </button>
                  <button
                    onClick={() => handleAuthClick('signup')}
                    className="bg-primary-600 text-white px-3 py-2 rounded-lg hover:bg-primary-700 transition-colors"
                  >
                    Sign Up
                  </button>
                </div>
              )}
              
              <a
                href="https://github.com/Kibet-g/PassResume"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Github className="h-5 w-5" />
                <span className="hidden sm:inline">GitHub</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      {currentStep === 1 && (
        <section className="py-20">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="animate-fade-in">
              <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
                ATS Resume
                <span className="text-primary-600"> Optimizer</span>
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Simple 3-step process: Upload your resume, get your ATS score, and download an improved version.
              </p>
              
              {/* Simple Features */}
              <div className="grid md:grid-cols-3 gap-8 mb-12">
                <div className="flex flex-col items-center p-6">
                  <div className="bg-primary-100 p-4 rounded-full mb-4">
                    <Upload className="h-10 w-10 text-primary-600" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">1. Upload</h3>
                  <p className="text-gray-600">
                    Upload your resume in PDF or DOCX format
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-blue-100 p-4 rounded-full mb-4">
                    <Target className="h-10 w-10 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">2. Analyze</h3>
                  <p className="text-gray-600">
                    Get your ATS compatibility percentage score
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-green-100 p-4 rounded-full mb-4">
                    <Download className="h-10 w-10 text-green-600" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">3. Download</h3>
                  <p className="text-gray-600">
                    Get your improved, ATS-friendly resume
                  </p>
                </div>
              </div>

              {/* Stats Display */}
              {appStats && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mb-8">
                  <h3 className="text-lg font-semibold mb-4">Platform Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{appStats.total_resumes}</div>
                      <div className="text-sm text-gray-600">Resumes Optimized</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{appStats.average_ats_score}%</div>
                      <div className="text-sm text-gray-600">Avg ATS Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">{appStats.average_user_score}%</div>
                      <div className="text-sm text-gray-600">Success Rate</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-4">
            {[1, 2, 3].map((step) => (
              <div key={step} className="flex items-center">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-sm font-medium ${
                    currentStep >= step
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {step}
                </div>
                {step < 3 && (
                  <div
                    className={`w-16 h-1 mx-2 ${
                      currentStep > step ? 'bg-primary-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-center mt-4">
            <div className="text-center">
              <p className="text-lg font-medium text-gray-900">
                {currentStep === 1 && 'Upload Your Resume'}
                {currentStep === 2 && 'Analyze ATS Score'}
                {currentStep === 3 && 'Download Improved Resume'}
              </p>
              <p className="text-sm text-gray-500">
                {currentStep === 1 && 'Upload your resume file to get started'}
                {currentStep === 2 && 'Click to analyze your resume\'s ATS compatibility'}
                {currentStep === 3 && 'Download your optimized, ATS-friendly resume'}
              </p>
            </div>
          </div>
        </div>

        {/* Step Content */}
        <div className="animate-slide-up">
          {currentStep === 1 && (
            <FileUpload onUploadSuccess={handleResumeUpload} />
          )}
          
          {currentStep === 2 && resumeData && (
            <div className="card max-w-2xl mx-auto text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Ready to Analyze</h2>
              <p className="text-gray-600 mb-6">
                Your resume has been uploaded successfully. Click the button below to analyze your ATS compatibility score.
              </p>
              
              <div className="bg-blue-50 rounded-lg p-4 mb-6">
                <p className="text-sm text-blue-800">
                  <strong>File:</strong> {resumeData.file_name || 'Resume'} <br />
                  <strong>Status:</strong> Ready for analysis
                </p>
              </div>

              <button
                onClick={handleAnalyzeResume}
                disabled={isAnalyzing}
                className="bg-primary-600 text-white px-8 py-4 rounded-lg font-medium hover:bg-primary-700 transition-all duration-200 flex items-center gap-3 mx-auto text-lg disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Analyzing Resume...
                  </>
                ) : (
                  <>
                    <Target className="h-5 w-5" />
                    Analyze Your Resume
                  </>
                )}
              </button>
            </div>
          )}
          
          {currentStep === 3 && analysisResults && (
            <div className="card max-w-2xl mx-auto text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">ATS Analysis Results</h2>
              
              {/* ATS Score Display */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 mb-6">
                <div className="text-6xl font-bold mb-2">
                  <span className={analysisResults.ats_score >= 70 ? 'text-green-600' : 'text-orange-600'}>
                    {analysisResults.ats_score}%
                  </span>
                </div>
                <p className="text-lg text-gray-700 mb-2">ATS Compatibility Score</p>
                <p className={`text-sm font-medium ${analysisResults.ats_score >= 70 ? 'text-green-600' : 'text-orange-600'}`}>
                  {analysisResults.ats_score >= 70 ? '✅ ATS Friendly' : '⚠️ Needs Improvement'}
                </p>
              </div>

              {/* Improvement Message */}
              {analysisResults.auto_improved && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <p className="font-medium text-green-800">Auto-Improvement Applied</p>
                  </div>
                  <p className="text-sm text-green-700">
                    Your resume was automatically improved from {analysisResults.original_score}% to {analysisResults.ats_score}%
                  </p>
                </div>
              )}

              {/* Generate Button */}
              <button
                onClick={handleGenerateUpdatedResume}
                disabled={isGenerating}
                className="bg-green-600 text-white px-8 py-4 rounded-lg font-medium hover:bg-green-700 transition-all duration-200 flex items-center gap-3 mx-auto text-lg disabled:opacity-50 mb-6"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Generating Resume...
                  </>
                ) : (
                  <>
                    <Download className="h-5 w-5" />
                    Generate Updated ATS-Friendly Resume
                  </>
                )}
              </button>

              {/* Start Over Button */}
              <button
                onClick={resetProcess}
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                Start Over with New Resume
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-600 text-sm">
              Built with ❤️ for job seekers. 
              <a 
                href="https://github.com/Kibet-g/PassResume" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 ml-1"
              >
                Open source on GitHub
              </a>
            </p>
          </div>
        </div>
      </footer>

      {/* Authentication Modal */}
      {showAuthModal && (
        <AuthModal
          mode={authModalMode}
          onClose={() => setShowAuthModal(false)}
          onSwitchMode={(mode) => setAuthModalMode(mode)}
        />
      )}

      {/* User Dashboard */}
      {showDashboard && (
        <UserDashboard onClose={() => setShowDashboard(false)} />
      )}
    </div>
  )
}