'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import axios from 'axios'

interface FileUploadProps {
  onUploadSuccess: (data: any) => void
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [autoImproved, setAutoImproved] = useState(false)
  const [improvementDetails, setImprovementDetails] = useState<any>(null)
  
  const { token, isAuthenticated } = useAuth()

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)
    setUploadSuccess(false)
    setAutoImproved(false)
    setImprovementDetails(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const headers: any = {
        'Content-Type': 'multipart/form-data',
      }
      
      // Add authentication header if user is logged in
      if (token) {
        headers.Authorization = `Bearer ${token}`
      }

      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000'}/api/upload`, formData, {
        headers,
      })

      setUploadSuccess(true)
      
      // Check if auto-improvement was applied
      if (response.data.auto_improved) {
        setAutoImproved(true)
        setImprovementDetails(response.data)
      }
      
      setTimeout(() => {
        onUploadSuccess(response.data)
      }, 2000) // Increased delay to show improvement message
    } catch (error: any) {
      setUploadError(
        error.response?.data?.error || 'Failed to upload file. Please try again.'
      )
    } finally {
      setIsUploading(false)
    }
  }, [onUploadSuccess, token])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  })

  return (
    <div className="card max-w-2xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Resume</h2>
        <p className="text-gray-600">
          Upload your resume in PDF or DOCX format to get started with the ATS analysis
        </p>
        {isAuthenticated && (
          <p className="text-sm text-green-600 mt-2">
            ✅ Logged in - Your resume will be saved to your history
          </p>
        )}
      </div>

      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive && !isDragReject ? 'border-primary-400 bg-primary-50' : ''}
          ${isDragReject ? 'border-red-400 bg-red-50' : ''}
          ${!isDragActive ? 'border-gray-300 hover:border-gray-400' : ''}
          ${isUploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {isUploading ? (
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          ) : uploadSuccess ? (
            <CheckCircle className="h-12 w-12 text-success-600" />
          ) : (
            <Upload className="h-12 w-12 text-gray-400" />
          )}
          
          <div>
            {isUploading ? (
              <p className="text-lg font-medium text-gray-900">Uploading and parsing...</p>
            ) : uploadSuccess ? (
              <p className="text-lg font-medium text-success-600">Upload successful!</p>
            ) : isDragActive ? (
              <p className="text-lg font-medium text-primary-600">Drop your resume here</p>
            ) : (
              <div>
                <p className="text-lg font-medium text-gray-900">
                  Drag and drop your resume here
                </p>
                <p className="text-sm text-gray-500 mt-1">or click to browse files</p>
              </div>
            )}
          </div>
          
          {!isUploading && !uploadSuccess && (
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center space-x-1">
                <FileText className="h-4 w-4" />
                <span>PDF, DOCX</span>
              </div>
              <div>Max 10MB</div>
            </div>
          )}
        </div>
      </div>

      {uploadError && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <p className="text-red-800 font-medium">Upload Error</p>
          </div>
          <p className="text-red-700 text-sm mt-1">{uploadError}</p>
        </div>
      )}

      {uploadSuccess && (
        <div className="mt-4 space-y-3">
          <div className="p-4 bg-success-50 border border-success-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-success-600" />
              <p className="text-success-800 font-medium">Resume uploaded successfully!</p>
            </div>
            <p className="text-success-700 text-sm mt-1">
              {autoImproved ? 'Resume analyzed and automatically improved!' : 'Proceeding to the next step...'}
            </p>
          </div>

          {autoImproved && improvementDetails && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-5 w-5 text-blue-600" />
                <p className="text-blue-800 font-medium">Auto-Improvement Applied!</p>
              </div>
              <div className="text-sm text-blue-700 space-y-1">
                <p>
                  <strong>ATS Score:</strong> {improvementDetails.original_ats_score} → {improvementDetails.final_ats_score} 
                  <span className="text-green-600 font-medium"> (+{improvementDetails.score_improvement})</span>
                </p>
                {improvementDetails.improvements_made && improvementDetails.improvements_made.length > 0 && (
                  <div>
                    <p className="font-medium mt-2">Improvements made:</p>
                    <ul className="list-disc list-inside ml-2 space-y-1">
                      {improvementDetails.improvements_made.map((improvement: string, index: number) => (
                        <li key={index}>{improvement}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-2">What happens next?</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Your resume will be parsed and analyzed for ATS compliance</li>
          <li>• We'll calculate an initial ATS score based on formatting and structure</li>
          <li>• If your ATS score is below 70, we'll automatically apply improvements</li>
          <li>• Auto-improvements include: professional formatting, bullet points, missing sections, and ATS keywords</li>
          <li>• You can then add a job description for keyword matching analysis</li>
          <li>• Use the AI Resume Editor for further manual customizations</li>
        </ul>
      </div>
    </div>
  )
}