'use client'

import { useState } from 'react'
import { Upload, FileText, Target, Zap, Github, Star } from 'lucide-react'
import FileUpload from './components/FileUpload'
import JobDescriptionInput from './components/JobDescriptionInput'
import ResultsDisplay from './components/ResultsDisplay'

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1)
  const [resumeData, setResumeData] = useState<any>(null)
  const [jobDescription, setJobDescription] = useState('')
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleResumeUpload = (data: any) => {
    setResumeData(data)
    setCurrentStep(2)
  }

  const handleAnalysis = (results: any) => {
    setAnalysisResults(results)
    setCurrentStep(3)
  }

  const resetProcess = () => {
    setCurrentStep(1)
    setResumeData(null)
    setJobDescription('')
    setAnalysisResults(null)
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
                <p className="text-sm text-gray-600">Free ATS Resume Checker</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
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
                Beat the ATS,
                <span className="text-primary-600"> Land the Job</span>
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Free AI-powered resume auditing tool that ensures ATS compliance and increases your chances of landing interviews.
              </p>
              
              {/* Features */}
              <div className="grid md:grid-cols-3 gap-8 mb-12">
                <div className="flex flex-col items-center p-6">
                  <div className="bg-primary-100 p-3 rounded-full mb-4">
                    <Upload className="h-8 w-8 text-primary-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Upload & Parse</h3>
                  <p className="text-gray-600 text-sm">
                    Drag and drop your PDF or DOCX resume for instant parsing
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-success-100 p-3 rounded-full mb-4">
                    <Target className="h-8 w-8 text-success-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">ATS Analysis</h3>
                  <p className="text-gray-600 text-sm">
                    Get comprehensive ATS compliance score and feedback
                  </p>
                </div>
                <div className="flex flex-col items-center p-6">
                  <div className="bg-warning-100 p-3 rounded-full mb-4">
                    <Zap className="h-8 w-8 text-warning-600" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">AI Suggestions</h3>
                  <p className="text-gray-600 text-sm">
                    Receive personalized improvement recommendations
                  </p>
                </div>
              </div>
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
                {currentStep === 3 && 'View Results'}
              </p>
              <p className="text-xs text-gray-500">
                {currentStep === 1 && 'Upload your resume to get started'}
                {currentStep === 2 && 'Paste the job description for better analysis'}
                {currentStep === 3 && 'Review your ATS score and suggestions'}
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
            <ResultsDisplay
              results={analysisResults}
              onStartOver={resetProcess}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-600 text-sm">
              Built with ❤️ for the job-seeking community. 
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
    </div>
  )
}