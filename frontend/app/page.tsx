'use client'

import { useState, useEffect } from 'react'
import { Upload, FileText, Target, Zap, Star, Brain, Shield, Download, TrendingUp, Database, CheckCircle, User, LogIn, Wand2, Scan, ArrowRight, Sparkles, BarChart3, FileCheck, Rocket } from 'lucide-react'
import FileUpload from './components/FileUpload'
import AuthModal from './components/AuthModal'
import UserDashboard from './components/UserDashboard'
import AIScanner from './components/AIScanner'
import ResumeRefiner from './components/ResumeRefiner'
import PDFExport from './components/PDFExport'
import AIDashboard from './components/AIDashboard'
import AdminDashboard from './components/AdminDashboard'
import AdvancedAIProcessor from './components/AdvancedAIProcessor'
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
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-black/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <Rocket className="h-5 w-5 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-white">passResume</h1>
                <p className="text-xs text-gray-400">AI-powered resume optimization</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {appStats && (
                <div className="hidden md:flex items-center space-x-4 text-sm">
                  <div className="flex items-center space-x-2 px-3 py-1 bg-gray-900 rounded-full">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-gray-300">{appStats.total_resumes} optimized</span>
                  </div>
                  <div className="flex items-center space-x-2 px-3 py-1 bg-gray-900 rounded-full">
                    <BarChart3 className="h-3 w-3 text-purple-400" />
                    <span className="text-gray-300">{appStats.average_ats_score}% avg score</span>
                  </div>
                </div>
              )}
              
              {/* Authentication Section */}
              {loading ? (
                <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
              ) : isAuthenticated ? (
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => setShowDashboard(true)}
                    className="flex items-center space-x-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg transition-all duration-200 border border-gray-700"
                  >
                    <User className="h-4 w-4" />
                    <span className="hidden sm:inline">Dashboard</span>
                  </button>
                  <div className="text-sm text-gray-400">
                    {user?.full_name?.split(' ')[0]}
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => handleAuthClick('login')}
                    className="text-gray-300 hover:text-white transition-colors px-4 py-2"
                  >
                    Sign In
                  </button>
                  <button
                    onClick={() => handleAuthClick('signup')}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-4 py-2 rounded-lg transition-all duration-200"
                  >
                    Get Started
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-24 overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-pink-900/20"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,119,198,0.1),transparent_50%)]"></div>
        
        <div className="relative max-w-6xl mx-auto px-6 lg:px-8 text-center">
          <div className="space-y-8">
            {/* Badge */}
            <div className="inline-flex items-center space-x-2 bg-gray-900/50 border border-gray-700 rounded-full px-4 py-2 text-sm">
              <Sparkles className="h-4 w-4 text-purple-400" />
              <span className="text-gray-300">AI-Powered Resume Optimization</span>
            </div>
            
            {/* Main heading */}
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
              <span className="text-white">Transform your</span>
              <br />
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
                resume instantly
              </span>
            </h1>
            
            {/* Subtitle */}
            <p className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              Upload, analyze, and optimize your resume with advanced AI. 
              Get past ATS systems and land your dream job.
            </p>
            
            {/* CTA Button */}
            <div className="pt-4">
              <button
                onClick={() => setActiveTab('upload')}
                className="group inline-flex items-center space-x-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 rounded-xl text-lg font-medium transition-all duration-200 shadow-lg hover:shadow-purple-500/25"
              >
                <Upload className="h-5 w-5" />
                <span>Start Optimizing</span>
                <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
            
            {/* Process Steps */}
            <div className="pt-16">
              <div className="grid md:grid-cols-4 gap-8">
                <div className="group">
                  <div className="relative mb-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-200">
                      <Upload className="h-8 w-8 text-purple-400" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-xs font-bold text-white">
                      1
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Upload</h3>
                  <p className="text-gray-400 text-sm">
                    Drop your resume file and let our AI analyze it
                  </p>
                </div>
                
                <div className="group">
                  <div className="relative mb-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-200">
                      <Scan className="h-8 w-8 text-blue-400" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-xs font-bold text-white">
                      2
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Analyze</h3>
                  <p className="text-gray-400 text-sm">
                    Get detailed ATS compatibility and improvement insights
                  </p>
                </div>
                
                <div className="group">
                  <div className="relative mb-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-200">
                      <Wand2 className="h-8 w-8 text-green-400" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-xs font-bold text-white">
                      3
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Optimize</h3>
                  <p className="text-gray-400 text-sm">
                    AI-powered content enhancement and keyword optimization
                  </p>
                </div>
                
                <div className="group">
                  <div className="relative mb-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-orange-500/20 to-orange-600/20 border border-orange-500/30 rounded-2xl flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-200">
                      <Download className="h-8 w-8 text-orange-400" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-orange-500 rounded-full flex items-center justify-center text-xs font-bold text-white">
                      4
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Export</h3>
                  <p className="text-gray-400 text-sm">
                    Download your optimized resume in professional format
                  </p>
                </div>
              </div>
            </div>

            {/* Stats Display */}
            {appStats && (
              <div className="pt-16">
                <div className="bg-gray-900/50 border border-gray-700 rounded-2xl p-8 backdrop-blur-sm">
                  <h3 className="text-lg font-semibold text-white mb-6">Platform Impact</h3>
                  <div className="grid grid-cols-3 gap-8">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-purple-400 mb-1">{appStats.total_resumes}</div>
                      <div className="text-sm text-gray-400">Resumes Optimized</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-400 mb-1">{appStats.average_ats_score}%</div>
                      <div className="text-sm text-gray-400">Average ATS Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-pink-400 mb-1">{appStats.average_user_score}%</div>
                      <div className="text-sm text-gray-400">Success Rate</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 lg:px-8 pb-24">
        <div className="bg-gray-900/50 border border-gray-700 rounded-2xl backdrop-blur-sm overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <div className="border-b border-gray-700 bg-gray-900/30">
              <TabsList className="grid w-full grid-cols-6 bg-transparent border-0 p-0">
                <TabsTrigger 
                  value="upload" 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-300 data-[state=active]:border-b-2 data-[state=active]:border-purple-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0"
                >
                  <Upload className="h-4 w-4" />
                  <span className="hidden sm:inline">Upload</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="analyze" 
                  disabled={!resumeData} 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-blue-500/20 data-[state=active]:text-blue-300 data-[state=active]:border-b-2 data-[state=active]:border-blue-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0 disabled:opacity-50"
                >
                  <Target className="h-4 w-4" />
                  <span className="hidden sm:inline">Analyze</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="advanced-ai" 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-300 data-[state=active]:border-b-2 data-[state=active]:border-cyan-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0"
                >
                  <Brain className="h-4 w-4" />
                  <span className="hidden sm:inline">AI Pro</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="scan" 
                  disabled={!resumeText} 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-green-500/20 data-[state=active]:text-green-300 data-[state=active]:border-b-2 data-[state=active]:border-green-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0 disabled:opacity-50"
                >
                  <Scan className="h-4 w-4" />
                  <span className="hidden sm:inline">Scan</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="enhance" 
                  disabled={!resumeText} 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-pink-500/20 data-[state=active]:text-pink-300 data-[state=active]:border-b-2 data-[state=active]:border-pink-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0 disabled:opacity-50"
                >
                  <Wand2 className="h-4 w-4" />
                  <span className="hidden sm:inline">Enhance</span>
                </TabsTrigger>
                <TabsTrigger 
                  value="export" 
                  disabled={!resumeText} 
                  className="flex items-center gap-2 py-4 px-6 data-[state=active]:bg-orange-500/20 data-[state=active]:text-orange-300 data-[state=active]:border-b-2 data-[state=active]:border-orange-500 text-gray-400 hover:text-gray-300 transition-all duration-200 rounded-none border-0 disabled:opacity-50"
                >
                  <Download className="h-4 w-4" />
                  <span className="hidden sm:inline">Export</span>
                </TabsTrigger>
              </TabsList>
            </div>

            <div className="p-8">
              <TabsContent value="upload" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">Upload Your Resume</h3>
                  <p className="text-gray-400">Start by uploading your resume file to begin the AI optimization process</p>
                </div>
                <FileUpload onUploadSuccess={handleResumeUpload} />
              </TabsContent>

              <TabsContent value="analyze" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">ATS Analysis</h3>
                  <p className="text-gray-400">Analyze your resume's compatibility with Applicant Tracking Systems</p>
                </div>
                
                {resumeData && !analysisResults && (
                  <div className="bg-gray-800/50 border border-gray-600 rounded-xl p-8 text-center">
                    <h4 className="text-xl font-semibold text-white mb-4">Ready to Analyze</h4>
                    <p className="text-gray-400 mb-6">
                      Your resume has been uploaded successfully. Click the button below to analyze your ATS compatibility score.
                    </p>
                    
                    <div className="bg-gray-700/50 border border-gray-600 rounded-lg p-4 mb-6">
                      <p className="text-sm text-gray-300">
                        <strong>File:</strong> {resumeData.file_name || 'Resume'} <br />
                        <strong>Status:</strong> Ready for analysis
                      </p>
                    </div>

                    <Button
                      onClick={handleAnalyzeResume}
                      disabled={isAnalyzing}
                      size="lg"
                      className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white border-0"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                          Analyzing Resume...
                        </>
                      ) : (
                        <>
                          <Target className="h-5 w-5 mr-2" />
                          Analyze Your Resume
                        </>
                      )}
                    </Button>
                  </div>
                )}

                {analysisResults && (
                  <div className="bg-gray-800/50 border border-gray-600 rounded-xl p-8 text-center">
                    <h4 className="text-2xl font-bold text-white mb-6">ATS Analysis Results</h4>
                    
                    {/* ATS Score Display */}
                    <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-600 rounded-xl p-8 mb-6">
                      <div className="text-6xl font-bold mb-2">
                        <span className={analysisResults.ats_score >= 70 ? 'text-green-400' : 'text-orange-400'}>
                          {analysisResults.ats_score}%</span>
                      </div>
                      <p className="text-lg text-gray-400 mb-2">ATS Compatibility Score</p>
                      <Badge 
                        variant={analysisResults.ats_score >= 70 ? 'default' : 'secondary'}
                        className={analysisResults.ats_score >= 70 ? 'bg-green-500/20 text-green-300 border-green-500/30' : 'bg-orange-500/20 text-orange-300 border-orange-500/30'}
                      >
                        {analysisResults.ats_score >= 70 ? '✅ ATS Friendly' : '⚠️ Needs Improvement'}
                      </Badge>
                    </div>

                    {/* Improvement Message */}
                    {analysisResults.auto_improved && (
                      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 mb-6">
                        <div className="flex items-center justify-center gap-2 mb-2">
                          <CheckCircle className="h-5 w-5 text-green-400" />
                          <p className="font-medium text-green-300">Auto-Improvement Applied</p>
                        </div>
                        <p className="text-sm text-green-400">
                          Your resume was automatically improved from {analysisResults.original_score}% to {analysisResults.ats_score}%
                        </p>
                      </div>
                    )}

                    <div className="flex gap-4 justify-center">
                      <Button
                        onClick={handleGenerateUpdatedResume}
                        disabled={isGenerating}
                        size="lg"
                        className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white border-0"
                      >
                        {isGenerating ? (
                          <>
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
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
                        className="bg-gray-700 hover:bg-gray-600 text-white border border-gray-600"
                        size="lg"
                      >
                        <Scan className="h-5 w-5 mr-2" />
                        Continue to AI Scan
                      </Button>
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="advanced-ai" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">Advanced AI Processing</h3>
                  <p className="text-gray-400">Deep AI analysis with formatting preservation and self-learning capabilities</p>
                </div>
                <AdvancedAIProcessor />
              </TabsContent>

              <TabsContent value="scan" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">AI Resume Scanner</h3>
                  <p className="text-gray-400">Get detailed AI analysis of each section of your resume</p>
                </div>
                <AIScanner 
                  resumeText={resumeText} 
                  onScanComplete={handleScanComplete}
                />
              </TabsContent>

              <TabsContent value="enhance" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">AI Resume Enhancement</h3>
                  <p className="text-gray-400">Let AI refine and optimize your resume content</p>
                </div>
                <ResumeRefiner 
                  resumeText={resumeText}
                  onRefinementComplete={handleRefinementComplete}
                />
              </TabsContent>

              <TabsContent value="export" className="space-y-6 mt-0">
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-bold text-white mb-3">Professional PDF Export</h3>
                  <p className="text-gray-400">Export your enhanced resume as a professional PDF</p>
                </div>
                <PDFExport 
                  resumeText={resumeText}
                  refinedResume={refinedResume}
                  onExportComplete={handleExportComplete}
                />
              </TabsContent>
            </div>
          </Tabs>

          {/* Reset Button */}
          {resumeData && (
            <div className="text-center mt-8 pt-6 border-t border-gray-700">
              <Button
                onClick={resetProcess}
                className="bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white border border-gray-600"
              >
                Start Over with New Resume
              </Button>
            </div>
          )}
        </div>

        {/* AI Dashboard for authenticated users */}
        {isAuthenticated && (
          <div className="mt-12">
            <AIDashboard />
          </div>
        )}

        {/* Admin Dashboard for system monitoring */}
        {isAuthenticated && (
          <div className="mt-12">
            <AdminDashboard />
          </div>
        )}
      </main>

      {/* Modals */}
      {showAuthModal && (
        <AuthModal
          isOpen={showAuthModal}
          initialMode={authModalMode}
          onClose={() => setShowAuthModal(false)}
        />
      )}

      {showDashboard && (
        <UserDashboard isOpen={showDashboard} onClose={() => setShowDashboard(false)} />
      )}
    </div>
  )
}