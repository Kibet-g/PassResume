'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react'
import axios from 'axios'

interface FileUploadProps {
  onUploadSuccess: (data: any) => void
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)
    setUploadSuccess(false)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setUploadSuccess(true)
      setTimeout(() => {
        onUploadSuccess(response.data)
      }, 1000)
    } catch (error: any) {
      setUploadError(
        error.response?.data?.error || 'Failed to upload file. Please try again.'
      )
    } finally {
      setIsUploading(false)
    }
  }, [onUploadSuccess])

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
        <div className="mt-4 p-4 bg-success-50 border border-success-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-success-600" />
            <p className="text-success-800 font-medium">Resume uploaded successfully!</p>
          </div>
          <p className="text-success-700 text-sm mt-1">
            Proceeding to the next step...
          </p>
        </div>
      )}

      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-2">What happens next?</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Your resume will be parsed and analyzed for ATS compliance</li>
          <li>• We'll extract key information like contact details, experience, and skills</li>
          <li>• You'll get an initial ATS score based on formatting and structure</li>
          <li>• Next, you can add a job description for keyword matching analysis</li>
        </ul>
      </div>
    </div>
  )
}