'use client'

import { useState, useEffect } from 'react'
import { Upload, FileText, Target, Zap, Star, Brain, Shield, Download, TrendingUp, Database, CheckCircle, User, LogIn, Wand2, Scan } from 'lucide-react'
import FileUpload from './components/FileUpload'
import AuthModal from './components/AuthModal'
import UserDashboard from './components/UserDashboard'
import AIScanner from './components/AIScanner'
import ResumeRefiner from './components/ResumeRefiner'
import PDFExport from './components/PDFExport'
import AIDashboard from './components/AIDashboard'
import { useAuth } from './contexts/AuthContext'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Badge } from './components/ui/badge'
import axios from 'axios'

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1)
  const [resumeData, setResumeData] = useState<any>(null)
  const [resumeText, setResumeText] = useState<string>('')
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [scanResults, setScanResults] = useState<any>(null)
  const [refinedResume, setRefinedResume] = useState<string>('')
  const [improvements, setImprovements] = useState<any[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [appStats, setAppStats] = useState<any>(null)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [showDashboard, setShowDashboard] = useState(false)
  const [authModalMode, setAuthModalMode] = useState<'login' | 'signup'>('login')
  const [activeTab, setActiveTab] = useState('upload')

  const { user, isAuthenticated, loading } = useAuth()

  // Fetch app statistics on load
  useEffect(() => {
    fetchAppStats()
  }, [])

  const fetchAppStats = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000'}/api/stats`)
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
    // Extract text from the uploaded resume for AI processing
    if (data.extracted_text) {
      setResumeText(data.extracted_text)
    }
    setCurrentStep(2)
    setActiveTab('analyze')
    fetchAppStats()
  }

  const handleAnalyzeResume = async () => {
    if (!resumeData) return
    
    setIsAnalyzing(true)
    try {
      // Get the current resume data from the database
      const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000'}/api/resume/${resumeData.resume_id}`)
      const resumeDetails = response.data
      
      setAnalysisResults({
        ats_score: resumeDetails.ats_score,
        ats_friendly: resumeDetails.ats_score >= 70,
        resume_id: resumeData.resume_id,
        auto_improved: resumeDetails.auto_improved || false,
        original_score: resumeDetails.original_ats_score || resumeDetails.ats_score
      })
      setCurrentStep(3)
      setActiveTab('enhance')
    } catch (error) {
      console.error('Error analyzing resume:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleScanComplete = (results: any) => {
    setScanResults(results)
  }

  const handleRefinementComplete = (refined: string, improvementList: any[]) => {
    setRefinedResume(refined)
    setImprovements(improvementList)
    setActiveTab('export')
  }

  const handleExportComplete = (filename: string) => {
    console.log('PDF exported:', filename)
  }

  const handleGenerateUpdatedResume = async () => {
    if (!resumeData) return
    
    setIsGenerating(true)
    try {
      // Call the auto-improve endpoint
      const response = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000'}/api/auto-improve-resume`, {
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
      setAnalysisResults((prev: any) => ({
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
    setResumeText('')
    setAnalysisResults(null)
    setScanResults(null)
    setRefinedResume('')
    setImprovements([])
    setActiveTab('upload')
  }

  const handleAuthClick = (mode: 'login' | 'signup') => {
    setAuthModalMode(mode)
    setShowAuthModal(true)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">PassResume AI</h1>
                <p className="text-sm text-gray-600">Intelligent Resume Enhancement Platform</p>
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
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
              ) : isAuthenticated ? (
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setShowDashboard(true)}
                    className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-3 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
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
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-3 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
                  >
                    Sign Up
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="animate-fade-in">
            <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              AI-Powered Resume
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"> Enhancement</span>
            </h2>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced AI technology that scans, analyzes, and refines your resume with intelligent insights and professional PDF export.
            </p>
            
            {/* Enhanced Features */}
            <div className="grid md:grid-cols-4 gap-6 mb-12">
              <div className="flex flex-col items-center p-6">
                <div className="bg-blue-100 p-4 rounded-full mb-4">
                  <Upload className="h-8 w-8 text-blue-600" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Upload</h3>
                <p className="text-gray-600 text-sm">
                  Upload your resume in multiple formats
                </p>
              </div>
              <div className="flex flex-col items-center p-6">
                <div className="bg-purple-100 p-4 rounded-full mb-4">
                  <Scan className="h-8 w-8 text-purple-600" />
                </div>
                <h3 className="text-lg font-semibold mb-2">AI Scan</h3>
                <p className="text-gray-600 text-sm">
                  Intelligent section analysis and quality scoring
                </p>
              </div>
              <div className="flex flex-col items-center p-6">
                <div className="bg-green-100 p-4 rounded-full mb-4">
                  <Wand2 className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold mb-2">AI Refine</h3>
                <p className="text-gray-600 text-sm">
                  Smart content enhancement and optimization
                </p>
              </div>
              <div className="flex flex-col items-center p-6">
                <div className="bg-orange-100 p-4 rounded-full mb-4">
                  <Download className="h-8 w-8 text-orange-600" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Export PDF</h3>
                <p className="text-gray-600 text-sm">
                  Professional PDF with multiple templates
                </p>
              </div>
            </div>

            {/* Stats Display */}
            {appStats && (
              <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-0 mb-8">
                <CardContent className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Platform Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{appStats.total_resumes}</div>
                      <div className="text-sm text-gray-600">Resumes Enhanced</div>
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
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
          <CardContent className="p-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-5 mb-8">
                <TabsTrigger value="upload" className="flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="analyze" disabled={!resumeData} className="flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Analyze
                </TabsTrigger>
                <TabsTrigger value="scan" disabled={!resumeText} className="flex items-center gap-2">
                  <Scan className="h-4 w-4" />
                  AI Scan
                </TabsTrigger>
                <TabsTrigger value="enhance" disabled={!resumeText} className="flex items-center gap-2">
                  <Wand2 className="h-4 w-4" />
                  Enhance
                </TabsTrigger>
                <TabsTrigger value="export" disabled={!resumeText} className="flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Export
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Resume</h3>
                  <p className="text-gray-600">Start by uploading your resume file to begin the AI enhancement process</p>
                </div>
                <FileUpload onUploadSuccess={handleResumeUpload} />
              </TabsContent>

              <TabsContent value="analyze" className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">ATS Analysis</h3>
                  <p className="text-gray-600">Analyze your resume's compatibility with Applicant Tracking Systems</p>
                </div>
                
                {resumeData && !analysisResults && (
                  <Card>
                    <CardContent className="text-center p-8">
                      <h4 className="text-xl font-semibold text-gray-900 mb-4">Ready to Analyze</h4>
                      <p className="text-gray-600 mb-6">
                        Your resume has been uploaded successfully. Click the button below to analyze your ATS compatibility score.
                      </p>
                      
                      <div className="bg-blue-50 rounded-lg p-4 mb-6">
                        <p className="text-sm text-blue-800">
                          <strong>File:</strong> {resumeData.file_name || 'Resume'} <br />
                          <strong>Status:</strong> Ready for analysis
                        </p>
                      </div>

                      <Button
                        onClick={handleAnalyzeResume}
                        disabled={isAnalyzing}
                        size="lg"
                        className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                            Analyzing Resume...
                          </>
                        ) : (
                          <>
                            <Target className="h-5 w-5 mr-2" />
                            Analyze Your Resume
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>
                )}

                {analysisResults && (
                  <Card>
                    <CardContent className="text-center p-8">
                      <h4 className="text-2xl font-bold text-gray-900 mb-6">ATS Analysis Results</h4>
                      
                      {/* ATS Score Display */}
                      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 mb-6">
                        <div className="text-6xl font-bold mb-2">
                          <span className={analysisResults.ats_score >= 70 ? 'text-green-600' : 'text-orange-600'}>
                            {analysisResults.ats_score}%
                          </span>
                        </div>
                        <p className="text-lg text-gray-700 mb-2">ATS Compatibility Score</p>
                        <Badge variant={analysisResults.ats_score >= 70 ? 'default' : 'secondary'}>
                          {analysisResults.ats_score >= 70 ? '✅ ATS Friendly' : '⚠️ Needs Improvement'}
                        </Badge>
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

                      <div className="flex gap-4 justify-center">
                        <Button
                          onClick={handleGenerateUpdatedResume}
                          disabled={isGenerating}
                          size="lg"
                          className="bg-green-600 hover:bg-green-700"
                        >
                          {isGenerating ? (
                            <>
                              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                              Generating Resume...
                            </>
                          ) : (
                            <>
                              <Download className="h-5 w-5 mr-2" />
                              Download Improved Resume
                            </>
                          )}
                        </Button>
                        
                        <Button
                          onClick={() => setActiveTab('scan')}
                          variant="outline"
                          size="lg"
                        >
                          <Scan className="h-5 w-5 mr-2" />
                          Continue to AI Scan
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="scan" className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">AI Resume Scanner</h3>
                  <p className="text-gray-600">Get detailed AI analysis of each section of your resume</p>
                </div>
                <AIScanner 
                  resumeText={resumeText} 
                  onScanComplete={handleScanComplete}
                />
              </TabsContent>

              <TabsContent value="enhance" className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">AI Resume Enhancement</h3>
                  <p className="text-gray-600">Let AI refine and optimize your resume content</p>
                </div>
                <ResumeRefiner 
                  resumeText={resumeText}
                  onRefinementComplete={handleRefinementComplete}
                />
              </TabsContent>

              <TabsContent value="export" className="space-y-6">
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">Professional PDF Export</h3>
                  <p className="text-gray-600">Export your enhanced resume as a professional PDF</p>
                </div>
                <PDFExport 
                  resumeText={resumeText}
                  refinedResume={refinedResume}
                  onExportComplete={handleExportComplete}
                />
              </TabsContent>
            </Tabs>

            {/* Reset Button */}
            {resumeData && (
              <div className="text-center mt-8">
                <Button
                  onClick={resetProcess}
                  variant="outline"
                  className="text-gray-600 hover:text-gray-900"
                >
                  Start Over with New Resume
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI Dashboard for authenticated users */}
        {isAuthenticated && (
          <div className="mt-12">
            <AIDashboard />
          </div>
        )}
      </main>

      {/* Modals */}
      {showAuthModal && (
        <AuthModal
          mode={authModalMode}
          onClose={() => setShowAuthModal(false)}
        />
      )}

      {showDashboard && (
        <UserDashboard onClose={() => setShowDashboard(false)} />
      )}
    </div>
  )
}