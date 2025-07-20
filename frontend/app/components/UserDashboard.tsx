'use client'

import React, { useState, useEffect } from 'react'
import { FileText, Calendar, TrendingUp, Download, Eye, X, BarChart3 } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import axios from 'axios'

interface Resume {
  id: number
  file_name: string
  ats_score: number
  upload_time: string
  auto_improved: boolean
  original_ats_score?: number
}

interface UserStats {
  total_resumes: number
  average_score: number
  best_score: number
  improvement_rate: number
}

interface UserDashboardProps {
  isOpen: boolean
  onClose: () => void
}

const UserDashboard: React.FC<UserDashboardProps> = ({ isOpen, onClose }) => {
  const { user, token, logout } = useAuth()
  const [resumes, setResumes] = useState<Resume[]>([])
  const [stats, setStats] = useState<UserStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedResume, setSelectedResume] = useState<Resume | null>(null)

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000'

  useEffect(() => {
    if (isOpen && token) {
      fetchUserData()
    }
  }, [isOpen, token])

  const fetchUserData = async () => {
    try {
      setLoading(true)
      
      // Fetch user resumes
      const resumesResponse = await axios.get(`${API_BASE_URL}/api/user/resumes`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setResumes(resumesResponse.data.resumes)

      // Fetch user stats
      const statsResponse = await axios.get(`${API_BASE_URL}/api/user/stats`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setStats(statsResponse.data)
    } catch (error) {
      console.error('Error fetching user data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadResume = async (resumeId: number, fileName: string) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/resume/${resumeId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      
      const resumeData = response.data
      const blob = new Blob([resumeData.parsed_text], { type: 'text/plain' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${fileName.replace(/\.[^/.]+$/, '')}_ats_${resumeData.ats_score}.txt`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading resume:', error)
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 70) return 'text-blue-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'bg-green-100'
    if (score >= 70) return 'bg-blue-100'
    if (score >= 60) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">My Dashboard</h2>
            <p className="text-gray-600">Welcome back, {user?.full_name}</p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={logout}
              className="text-gray-500 hover:text-gray-700 px-3 py-1 rounded text-sm"
            >
              Sign Out
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              <span className="ml-2 text-gray-600">Loading your data...</span>
            </div>
          ) : (
            <>
              {/* Stats Section */}
              {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="bg-blue-50 rounded-lg p-4 text-center">
                    <FileText className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-blue-600">{stats.total_resumes}</div>
                    <div className="text-sm text-gray-600">Total Resumes</div>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 text-center">
                    <BarChart3 className="h-8 w-8 text-green-600 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-green-600">{stats.average_score}%</div>
                    <div className="text-sm text-gray-600">Avg Score</div>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 text-center">
                    <TrendingUp className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-purple-600">{stats.best_score}%</div>
                    <div className="text-sm text-gray-600">Best Score</div>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-4 text-center">
                    <TrendingUp className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-orange-600">{stats.improvement_rate}%</div>
                    <div className="text-sm text-gray-600">Improvement Rate</div>
                  </div>
                </div>
              )}

              {/* Resume History */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Resume History</h3>
                
                {resumes.length === 0 ? (
                  <div className="text-center py-12">
                    <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">No resumes uploaded yet</p>
                    <p className="text-sm text-gray-500">Upload your first resume to get started!</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {resumes.map((resume) => (
                      <div key={resume.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3">
                              <FileText className="h-5 w-5 text-gray-400" />
                              <div>
                                <h4 className="font-medium text-gray-900">{resume.file_name}</h4>
                                <div className="flex items-center space-x-4 text-sm text-gray-500">
                                  <span className="flex items-center">
                                    <Calendar className="h-4 w-4 mr-1" />
                                    {formatDate(resume.upload_time)}
                                  </span>
                                  {resume.auto_improved && (
                                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">
                                      Auto-improved
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-4">
                            {/* Score Display */}
                            <div className="text-center">
                              <div className={`text-lg font-bold ${getScoreColor(resume.ats_score)}`}>
                                {resume.ats_score}%
                              </div>
                              {resume.auto_improved && resume.original_ats_score && (
                                <div className="text-xs text-gray-500">
                                  from {resume.original_ats_score}%
                                </div>
                              )}
                            </div>
                            
                            {/* Actions */}
                            <div className="flex space-x-2">
                              <button
                                onClick={() => setSelectedResume(resume)}
                                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                                title="View details"
                              >
                                <Eye className="h-4 w-4" />
                              </button>
                              <button
                                onClick={() => handleDownloadResume(resume.id, resume.file_name)}
                                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                                title="Download resume"
                              >
                                <Download className="h-4 w-4" />
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Resume Detail Modal */}
      {selectedResume && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-60 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Resume Details</h3>
              <button
                onClick={() => setSelectedResume(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700">File Name</label>
                  <p className="text-gray-900">{selectedResume.file_name}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Upload Date</label>
                  <p className="text-gray-900">{formatDate(selectedResume.upload_time)}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">ATS Score</label>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getScoreBgColor(selectedResume.ats_score)} ${getScoreColor(selectedResume.ats_score)}`}>
                    {selectedResume.ats_score}%
                  </div>
                </div>
                {selectedResume.auto_improved && (
                  <div>
                    <label className="text-sm font-medium text-gray-700">Improvement</label>
                    <p className="text-green-600">
                      Auto-improved from {selectedResume.original_ats_score}% to {selectedResume.ats_score}%
                      <span className="text-gray-600 ml-2">
                        (+{selectedResume.ats_score - (selectedResume.original_ats_score || 0)} points)
                      </span>
                    </p>
                  </div>
                )}
              </div>
              <div className="mt-6 flex space-x-3">
                <button
                  onClick={() => handleDownloadResume(selectedResume.id, selectedResume.file_name)}
                  className="bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
                >
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </button>
                <button
                  onClick={() => setSelectedResume(null)}
                  className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default UserDashboard