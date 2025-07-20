'use client'

import { useState, useEffect } from 'react'
import { Upload, FileText, Target, Zap, Github, Star, Brain, Shield, Edit3, TrendingUp, Database } from 'lucide-react'
import FileUpload from './components/FileUpload'
import JobDescriptionInput from './components/JobDescriptionInput'
import ResultsDisplay from './components/ResultsDisplay'
import ResumeEditor from './components/ResumeEditor'

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1)
  const [resumeData, setResumeData] = useState<any>(null)
  const [jobDescription, setJobDescription] = useState('')
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showEditor, setShowEditor] = useState(false)
  const [appStats, setAppStats] = useState<any>(null)

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
    // Refresh stats after upload
    fetchAppStats()
  }

  const handleAnalysis = (results: any) => {
    setAnalysisResults(results)
    setCurrentStep(3)
  }

  const handleEditResume = () => {
    setShowEditor(true)
  }

  const handleSaveEditedResume = (editedText: string) => {
    // Update the resume data with edited text
    setResumeData(prev => ({
      ...prev,
      preview: editedText
    }))
    setShowEditor(false)
    // Optionally re-analyze with edited content
  }

  const resetProcess = () => {
    setCurrentStep(1)
    setResumeData(null)
    setJobDescription('')
    setAnalysisResults(null)
    setShowEditor(false)
  }

  if (showEditor && resumeData) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center space-x-3">
                <div className="bg-primary-600 p-2 rounded-lg">
                  <Edit3 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">Resume Editor</h1>
                  <p className="text-sm text-gray-600">AI-Powered Resume Enhancement</p>
                </div>
              </div>
              <button
                onClick={() => setShowEditor(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                Back to Analysis
              </button>
            </div>
          </div>
        </header>
        <ResumeEditor
          resumeId={resumeData.resume_id}
          originalText={resumeData.preview}
          onSave={handleSaveEditedResume}
        />
      </div>
    )
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
                <h1 className="text-xl font-bold text-gray-900">Open Resume Auditor</h1>
                <p className="text-sm text-gray-600">AI-Powered ATS Resume Checker</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {/* ML Status Indicator */}
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
              <a
                href="https://github.com/Kibet-g/PassResume"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Github className="h-5 w-5" />
                <span className="hidden sm:inline">GitHub</span>
              </a>
              <button className="flex items-center space-x-1 bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-medium">
                <Star className="h-4 w-4" />
                <span>Star</span>
              </button>
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
                AI-Powered Resume
                <span className="text-primary-600"> Optimization</span>
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Advanced ML-powered resume auditing with data integrity, learning from thousands of resumes, and intelligent editing suggestions.
              </p>
              
              {/* Enhanced Features */}
              <div className="grid md:grid-cols-4 gap-6 mb-12">
                <div className="flex flex-col items-center p-6">
                  <div className="bg-primary-100 p-3 rounded-full mb-4">
                    <Upload className="h-8 w-8 text-primary-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Smart Upload</h3>
                  <p className="text-gray-600 text-sm">
                    Secure upload with duplicate detection and data integrity
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-blue-100 p-3 rounded-full mb-4">
                    <Brain className="h-8 w-8 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">ML Analysis</h3>
                  <p className="text-gray-600 text-sm">
                    Machine learning powered by {appStats?.ml_training_samples || 0}+ resumes
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-green-100 p-3 rounded-full mb-4">
                    <Edit3 className="h-8 w-8 text-green-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">AI Editor</h3>
                  <p className="text-gray-600 text-sm">
                    Real-time editing with intelligent suggestions
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-purple-100 p-3 rounded-full mb-4">
                    <Shield className="h-8 w-8 text-purple-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Data Integrity</h3>
                  <p className="text-gray-600 text-sm">
                    Secure processing with hash verification
                  </p>
                </div>
              </div>

              {/* Stats Display */}
              {appStats && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mb-8">
                  <h3 className="text-lg font-semibold mb-4">Platform Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{appStats.total_resumes}</div>
                      <div className="text-sm text-gray-600">Resumes Analyzed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{appStats.average_ats_score}%</div>
                      <div className="text-sm text-gray-600">Avg ATS Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">{appStats.ml_training_samples}</div>
                      <div className="text-sm text-gray-600">ML Training Data</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">{appStats.average_user_score}%</div>
                      <div className="text-sm text-gray-600">User Satisfaction</div>
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
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
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
              <p className="text-sm font-medium text-gray-900">
                {currentStep === 1 && 'Upload Resume'}
                {currentStep === 2 && 'Add Job Description'}
                {currentStep === 3 && 'View Results & Edit'}
              </p>
              <p className="text-xs text-gray-500">
                {currentStep === 1 && 'Upload your resume for ML-powered analysis'}
                {currentStep === 2 && 'Paste job description for keyword optimization'}
                {currentStep === 3 && 'Review results and edit your resume'}
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
            <JobDescriptionInput
              resumeData={resumeData}
              jobDescription={jobDescription}
              setJobDescription={setJobDescription}
              onAnalysisComplete={handleAnalysis}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}
          
          {currentStep === 3 && analysisResults && (
            <div className="space-y-6">
              <ResultsDisplay
                results={analysisResults}
                onStartOver={resetProcess}
              />
              
              {/* Resume Editor Button */}
              <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 text-center">
                <h3 className="text-lg font-semibold mb-2 flex items-center justify-center gap-2">
                  <Edit3 className="h-5 w-5" />
                  Want to improve your resume?
                </h3>
                <p className="text-gray-600 mb-4">
                  Use our AI-powered editor to implement suggestions and optimize your resume in real-time.
                </p>
                <button
                  onClick={handleEditResume}
                  className="bg-gradient-to-r from-green-600 to-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:from-green-700 hover:to-blue-700 transition-all duration-200 flex items-center gap-2 mx-auto"
                >
                  <Edit3 className="h-4 w-4" />
                  Open Resume Editor
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-600 text-sm">
              Built with ❤️ and AI for the job-seeking community. 
              <a 
                href="https://github.com/Kibet-g/PassResume" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 ml-1"
              >
                Open source on GitHub
              </a>
            </p>
            <p className="text-xs text-gray-500 mt-2">
              Powered by Machine Learning • Data Integrity Verified • Privacy Protected
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}