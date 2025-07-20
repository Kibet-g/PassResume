'use client'

import { useState } from 'react'
import { FileText, Briefcase, ArrowRight, AlertCircle } from 'lucide-react'
import axios from 'axios'

interface JobDescriptionInputProps {
  resumeData: any
  jobDescription: string
  setJobDescription: (value: string) => void
  onAnalysisComplete: (results: any) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function JobDescriptionInput({
  resumeData,
  jobDescription,
  setJobDescription,
  onAnalysisComplete,
  isLoading,
  setIsLoading
}: JobDescriptionInputProps) {
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/analyze', {
        resume_id: resumeData.resume_id,
        job_description: jobDescription
      })

      onAnalysisComplete(response.data)
    } catch (error: any) {
      setError(
        error.response?.data?.error || 'Failed to analyze resume. Please try again.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleSkipJobDescription = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post('/api/analyze', {
        resume_id: resumeData.resume_id,
        job_description: ''
      })

      onAnalysisComplete(response.data)
    } catch (error: any) {
      setError(
        error.response?.data?.error || 'Failed to analyze resume. Please try again.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Resume Summary Card */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className="bg-success-100 p-2 rounded-lg">
            <FileText className="h-5 w-5 text-success-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Resume Uploaded</h3>
            <p className="text-sm text-gray-600">ATS Score: {resumeData.ats_score}%</p>
          </div>
        </div>
        
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">Preview:</h4>
          <p className="text-sm text-gray-700 leading-relaxed">
            {resumeData.text_preview}
          </p>
        </div>

        {resumeData.feedback && resumeData.feedback.length > 0 && (
          <div className="mt-4">
            <h4 className="font-medium text-gray-900 mb-2">Initial Feedback:</h4>
            <ul className="space-y-1">
              {resumeData.feedback.slice(0, 3).map((item: string, index: number) => (
                <li key={index} className="text-sm text-gray-600 flex items-start space-x-2">
                  <span className="text-warning-500 mt-1">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Job Description Input */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <div className="bg-primary-100 p-2 rounded-lg">
            <Briefcase className="h-5 w-5 text-primary-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Job Description (Optional)</h3>
            <p className="text-sm text-gray-600">
              Add the job description to get keyword matching analysis
            </p>
          </div>
        </div>

        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Paste the job description here to get keyword matching analysis and tailored suggestions..."
          className="textarea-field h-40"
          disabled={isLoading}
        />

        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Why add a job description?</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Get keyword matching analysis</li>
            <li>• Receive tailored suggestions for this specific role</li>
            <li>• See which important keywords you're missing</li>
            <li>• Improve your chances of passing ATS filters</li>
          </ul>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={handleAnalyze}
          disabled={isLoading || !jobDescription.trim()}
          className="btn-primary flex-1 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
          ) : (
            <>
              <span>Analyze with Job Description</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>

        <button
          onClick={handleSkipJobDescription}
          disabled={isLoading}
          className="btn-secondary flex-1 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span>Skip & Analyze Resume Only</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <p className="text-red-800 font-medium">Analysis Error</p>
          </div>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}
    </div>
  )
}