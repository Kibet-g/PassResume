'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../app/components/ui/card'
import { Button } from '../app/components/ui/button'
import { Input } from '../app/components/ui/input'
import { Badge } from '../app/components/ui/badge'
import { Progress } from '../app/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../app/components/ui/tabs'
import { 
  Search, 
  MapPin, 
  Clock, 
  DollarSign, 
  Building, 
  Star, 
  Heart, 
  ExternalLink,
  User,
  Briefcase,
  GraduationCap,
  Target,
  TrendingUp,
  Filter,
  Loader2,
  CheckCircle,
  AlertCircle,
  Brain,
  Lightbulb,
  Shield,
  Zap,
  BarChart3,
  Cpu,
  Sparkles,
  Rocket,
  Award,
  Users
} from 'lucide-react'

interface UserProfile {
  name: string
  email: string
  phone: string
  location: string
  skills: string[]
  experience_years: number
  education_level: string
  job_titles: string[]
  industries: string[]
  keywords: string[]
}

interface JobOpportunity {
  id?: number
  title: string
  company: string
  location: string
  description: string
  requirements: string
  salary_range: string
  job_type: string
  experience_level: string
  posted_date: string
  application_url: string
  match_score: number
}

interface UltraAIAnalysis {
  profile_insights: {
    personality_analysis: {
      personality_score: number
      leadership_potential: number
      innovation_index: number
      adaptability_score: number
      communication_style: string
      learning_velocity: number
      stress_tolerance: number
      team_compatibility: number
      career_trajectory: string
      market_value: number
    }
    cognitive_assessment: {
      analytical_thinking: number
      verbal_reasoning: number
      technical_reasoning: number
      problem_solving: number
      abstract_thinking: number
    }
    emotional_intelligence: number
    creativity_index: number
    technical_aptitude: Record<string, number>
    soft_skills_matrix: Record<string, number>
    personality_traits: Record<string, number>
    career_aspirations: string[]
    learning_preferences: Record<string, any>
    work_style_analysis: Record<string, any>
  }
  career_intelligence: {
    career_level: string
    preferred_roles: string[]
    salary_expectation: Record<string, number>
    work_preferences: Record<string, any>
    experience_years: number
  }
  skills_analysis: {
    technical_skills: any[]
    soft_skills: any[]
    certifications: string[]
    education: any[]
  }
  ai_powered_jobs: any[]
  ai_system_info: {
    analysis_depth: string
    ai_models_used: string[]
    features_enabled: string[]
  }
}

interface JobMatcherProps {
  resumeText: string
  isAuthenticated: boolean
}

export default function JobMatcher({ resumeText, isAuthenticated }: JobMatcherProps) {
  const [activeTab, setActiveTab] = useState('ultra-ai')
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null)
  const [jobRecommendations, setJobRecommendations] = useState<JobOpportunity[]>([])
  const [savedJobs, setSavedJobs] = useState<JobOpportunity[]>([])
  const [searchResults, setSearchResults] = useState<JobOpportunity[]>([])
  const [ultraAnalysis, setUltraAnalysis] = useState<UltraAIAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [profileLoading, setProfileLoading] = useState(false)
  const [message, setMessage] = useState('')
  
  // Search filters
  const [searchKeywords, setSearchKeywords] = useState('')
  const [searchLocation, setSearchLocation] = useState('')
  const [jobType, setJobType] = useState('')
  const [experienceLevel, setExperienceLevel] = useState('')

  // Extract user profile on component mount
  useEffect(() => {
    if (resumeText && isAuthenticated) {
      extractUserProfile()
    }
  }, [resumeText, isAuthenticated])

  const showMessage = (msg: string, type: 'success' | 'error' = 'success') => {
    setMessage(msg)
    setTimeout(() => setMessage(''), 5000)
  }

  const analyzeWithUltraAI = async () => {
    if (!resumeText) {
      showMessage('Please upload and analyze your resume first', 'error')
      return
    }

    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/ultra-ai/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ resume_text: resumeText })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setUltraAnalysis(data.ultra_ai_analysis)
        showMessage('Ultra AI analysis completed successfully!')
      } else {
        showMessage(data.error || 'Failed to complete Ultra AI analysis', 'error')
      }
    } catch (error) {
      console.error('Error with Ultra AI analysis:', error)
      showMessage('Failed to complete Ultra AI analysis', 'error')
    } finally {
      setLoading(false)
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getScoreBg = (score: number) => {
    if (score >= 0.8) return 'bg-green-100'
    if (score >= 0.6) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  const extractUserProfile = async () => {
    if (!resumeText) {
      showMessage('Please upload and analyze your resume first', 'error')
      return
    }

    setProfileLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/job-matching/profile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ resume_text: resumeText })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setUserProfile(data.profile)
        showMessage('Profile extracted successfully!')
      } else {
        showMessage(data.error || 'Failed to extract profile', 'error')
      }
    } catch (error) {
      console.error('Error extracting profile:', error)
      showMessage('Failed to extract profile', 'error')
    } finally {
      setProfileLoading(false)
    }
  }

  const getJobRecommendations = async () => {
    if (!resumeText) {
      showMessage('Please upload and analyze your resume first', 'error')
      return
    }

    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/job-matching/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          resume_text: resumeText,
          limit: 20
        })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setJobRecommendations(data.data.recommendations || [])
        showMessage(`Found ${data.data.recommendations?.length || 0} job recommendations!`)
      } else {
        showMessage(data.error || 'Failed to get recommendations', 'error')
      }
    } catch (error) {
      console.error('Error getting recommendations:', error)
      showMessage('Failed to get job recommendations', 'error')
    } finally {
      setLoading(false)
    }
  }

  const searchJobs = async () => {
    if (!searchKeywords.trim()) {
      showMessage('Please enter search keywords', 'error')
      return
    }

    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/job-matching/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          keywords: searchKeywords.split(',').map(k => k.trim()),
          location: searchLocation,
          job_type: jobType,
          experience_level: experienceLevel,
          limit: 20
        })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setSearchResults(data.jobs || [])
        showMessage(`Found ${data.jobs?.length || 0} jobs matching your criteria!`)
      } else {
        showMessage(data.error || 'Failed to search jobs', 'error')
      }
    } catch (error) {
      console.error('Error searching jobs:', error)
      showMessage('Failed to search jobs', 'error')
    } finally {
      setLoading(false)
    }
  }

  const saveJob = async (job: JobOpportunity) => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/job-matching/save-job', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ job_data: job })
      })

      const data = await response.json()
      if (data.status === 'success') {
        showMessage('Job saved successfully!')
        loadSavedJobs()
      } else {
        showMessage(data.error || 'Failed to save job', 'error')
      }
    } catch (error) {
      console.error('Error saving job:', error)
      showMessage('Failed to save job', 'error')
    }
  }

  const loadSavedJobs = async () => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/job-matching/saved-jobs', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      const data = await response.json()
      if (data.status === 'success') {
        setSavedJobs(data.saved_jobs || [])
      }
    } catch (error) {
      console.error('Error loading saved jobs:', error)
    }
  }

  const removeSavedJob = async (jobId: number) => {
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/job-matching/remove-saved-job/${jobId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      const data = await response.json()
      if (data.status === 'success') {
        showMessage('Job removed from saved list')
        loadSavedJobs()
      } else {
        showMessage(data.error || 'Failed to remove job', 'error')
      }
    } catch (error) {
      console.error('Error removing job:', error)
      showMessage('Failed to remove job', 'error')
    }
  }

  // Load saved jobs when switching to saved tab
  useEffect(() => {
    if (activeTab === 'saved' && isAuthenticated) {
      loadSavedJobs()
    }
  }, [activeTab, isAuthenticated])

  const JobCard = ({ job, showSaveButton = true, showRemoveButton = false }: { 
    job: JobOpportunity, 
    showSaveButton?: boolean,
    showRemoveButton?: boolean 
  }) => (
    <Card className="mb-4 hover:shadow-lg transition-shadow border-gray-700 bg-gray-800">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <CardTitle className="text-lg text-white mb-1">{job.title}</CardTitle>
            <CardDescription className="text-gray-300 flex items-center gap-2">
              <Building className="w-4 h-4" />
              {job.company}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="bg-blue-600 text-white">
              {Math.round(job.match_score)}% Match
            </Badge>
            {showSaveButton && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => saveJob(job)}
                className="text-gray-400 hover:text-red-400"
              >
                <Heart className="w-4 h-4" />
              </Button>
            )}
            {showRemoveButton && job.id && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeSavedJob(job.id!)}
                className="text-gray-400 hover:text-red-400"
              >
                <Heart className="w-4 h-4 fill-current" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2 text-sm text-gray-400">
            <span className="flex items-center gap-1">
              <MapPin className="w-3 h-3" />
              {job.location}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {job.job_type}
            </span>
            <span className="flex items-center gap-1">
              <GraduationCap className="w-3 h-3" />
              {job.experience_level}
            </span>
            {job.salary_range && (
              <span className="flex items-center gap-1">
                <DollarSign className="w-3 h-3" />
                {job.salary_range}
              </span>
            )}
          </div>
          
          <p className="text-gray-300 text-sm line-clamp-3">
            {job.description}
          </p>
          
          <div className="flex justify-between items-center pt-2">
            <span className="text-xs text-gray-500">
              Posted: {new Date(job.posted_date).toLocaleDateString()}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open(job.application_url, '_blank')}
              className="border-blue-600 text-blue-400 hover:bg-blue-600 hover:text-white"
            >
              Apply Now
              <ExternalLink className="w-3 h-3 ml-1" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  if (!isAuthenticated) {
    return (
      <Card className="border-gray-700 bg-gray-800">
        <CardContent className="p-8 text-center">
          <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Authentication Required</h3>
          <p className="text-gray-400">Please sign in to access job matching features.</p>
        </CardContent>
      </Card>
    )
  }

  if (!resumeText) {
    return (
      <Card className="border-gray-700 bg-gray-800">
        <CardContent className="p-8 text-center">
          <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Resume Required</h3>
          <p className="text-gray-400">Please upload and analyze your resume first to get personalized job recommendations.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Message Display */}
      {message && (
        <div className={`p-4 rounded-lg ${message.includes('Failed') || message.includes('error') ? 'bg-red-900 text-red-200' : 'bg-green-900 text-green-200'}`}>
          {message}
        </div>
      )}

      {/* User Profile Section */}
      {userProfile && (
        <Card className="border-gray-700 bg-gray-800">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <User className="w-5 h-5" />
              Your Profile
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-white mb-2">Skills</h4>
                <div className="flex flex-wrap gap-2">
                  {userProfile.skills.slice(0, 8).map((skill, index) => (
                    <Badge key={index} variant="secondary" className="bg-blue-600 text-white">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-white mb-2">Experience</h4>
                <p className="text-gray-300">{userProfile.experience_years} years</p>
                <p className="text-gray-400 text-sm">{userProfile.location}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800">
          <TabsTrigger value="ultra-ai" className="text-white">
            <Brain className="w-4 h-4 mr-2" />
            Ultra AI
          </TabsTrigger>
          <TabsTrigger value="recommendations" className="text-white">
            <Target className="w-4 h-4 mr-2" />
            Recommendations
          </TabsTrigger>
          <TabsTrigger value="search" className="text-white">
            <Search className="w-4 h-4 mr-2" />
            Search Jobs
          </TabsTrigger>
          <TabsTrigger value="saved" className="text-white">
            <Heart className="w-4 h-4 mr-2" />
            Saved Jobs
          </TabsTrigger>
        </TabsList>

        {/* Ultra AI Tab */}
        <TabsContent value="ultra-ai" className="space-y-4">
          <Card className="bg-gradient-to-r from-purple-500 to-blue-600 text-white border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <Brain className="h-8 w-8" />
                Ultra AI Career Intelligence
              </CardTitle>
              <CardDescription className="text-purple-100">
                Powered by cutting-edge AI models including BERT, GPT-2, and advanced ML algorithms
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={analyzeWithUltraAI}
                disabled={loading || !resumeText}
                className="bg-white text-purple-600 hover:bg-purple-50"
              >
                {loading ? (
                  <>
                    <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing with Ultra AI...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2 h-4 w-4" />
                    Start Ultra AI Analysis
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {ultraAnalysis && (
            <Tabs defaultValue="overview" className="space-y-4">
              <TabsList className="grid w-full grid-cols-5 bg-gray-800">
                <TabsTrigger value="overview" className="text-white">Overview</TabsTrigger>
                <TabsTrigger value="personality" className="text-white">Personality</TabsTrigger>
                <TabsTrigger value="cognitive" className="text-white">Cognitive</TabsTrigger>
                <TabsTrigger value="jobs" className="text-white">AI Jobs</TabsTrigger>
                <TabsTrigger value="insights" className="text-white">Insights</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2 text-white">
                        <Star className="h-4 w-4 text-yellow-500" />
                        Overall Score
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-400">
                        {Math.round(ultraAnalysis.profile_insights.personality_analysis.personality_score * 100)}%
                      </div>
                      <Progress 
                        value={ultraAnalysis.profile_insights.personality_analysis.personality_score * 100} 
                        className="mt-2"
                      />
                    </CardContent>
                  </Card>

                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2 text-white">
                        <TrendingUp className="h-4 w-4 text-blue-500" />
                        Leadership Potential
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-blue-400">
                        {Math.round(ultraAnalysis.profile_insights.personality_analysis.leadership_potential * 100)}%
                      </div>
                      <Progress 
                        value={ultraAnalysis.profile_insights.personality_analysis.leadership_potential * 100} 
                        className="mt-2"
                      />
                    </CardContent>
                  </Card>

                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2 text-white">
                        <Lightbulb className="h-4 w-4 text-orange-500" />
                        Innovation Index
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-orange-400">
                        {Math.round(ultraAnalysis.profile_insights.personality_analysis.innovation_index * 100)}%
                      </div>
                      <Progress 
                        value={ultraAnalysis.profile_insights.personality_analysis.innovation_index * 100} 
                        className="mt-2"
                      />
                    </CardContent>
                  </Card>

                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium flex items-center gap-2 text-white">
                        <DollarSign className="h-4 w-4 text-green-500" />
                        Market Value
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-400">
                        {Math.round(ultraAnalysis.profile_insights.personality_analysis.market_value * 100)}%
                      </div>
                      <Progress 
                        value={ultraAnalysis.profile_insights.personality_analysis.market_value * 100} 
                        className="mt-2"
                      />
                    </CardContent>
                  </Card>
                </div>

                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <BarChart3 className="h-5 w-5" />
                      Career Intelligence Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold mb-2 text-white">Career Trajectory</h4>
                        <Badge variant="outline" className="text-blue-400 border-blue-400">
                          {ultraAnalysis.profile_insights.personality_analysis.career_trajectory}
                        </Badge>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2 text-white">Communication Style</h4>
                        <Badge variant="outline" className="text-purple-400 border-purple-400">
                          {ultraAnalysis.profile_insights.personality_analysis.communication_style}
                        </Badge>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2 text-white">Career Level</h4>
                        <Badge variant="outline" className="text-green-400 border-green-400">
                          {ultraAnalysis.career_intelligence.career_level}
                        </Badge>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2 text-white">Experience</h4>
                        <Badge variant="outline" className="text-orange-400 border-orange-400">
                          {ultraAnalysis.career_intelligence.experience_years} years
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="personality" className="space-y-4">
                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Heart className="h-5 w-5 text-red-500" />
                      Personality Analysis
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                      AI-powered personality assessment based on resume language patterns
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {Object.entries(ultraAnalysis.profile_insights.personality_traits).map(([trait, score]) => (
                      <div key={trait} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="font-medium capitalize text-white">{trait.replace('_', ' ')}</span>
                          <span className={`font-bold ${getScoreColor(score as number)}`}>
                            {Math.round((score as number) * 100)}%
                          </span>
                        </div>
                        <Progress value={(score as number) * 100} className="h-2" />
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Users className="h-5 w-5 text-blue-500" />
                      Soft Skills Matrix
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {Object.entries(ultraAnalysis.profile_insights.soft_skills_matrix).map(([skill, score]) => (
                        <div key={skill} className={`p-3 rounded-lg ${getScoreBg(score as number)}`}>
                          <div className="font-medium capitalize">{skill.replace('_', ' ')}</div>
                          <div className={`text-lg font-bold ${getScoreColor(score as number)}`}>
                            {Math.round((score as number) * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="cognitive" className="space-y-4">
                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Cpu className="h-5 w-5 text-purple-500" />
                      Cognitive Assessment
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                      AI analysis of cognitive abilities and thinking patterns
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {Object.entries(ultraAnalysis.profile_insights.cognitive_assessment).map(([ability, score]) => (
                      <div key={ability} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="font-medium capitalize text-white">{ability.replace('_', ' ')}</span>
                          <span className={`font-bold ${getScoreColor(score as number)}`}>
                            {Math.round((score as number) * 100)}%
                          </span>
                        </div>
                        <Progress value={(score as number) * 100} className="h-2" />
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Emotional Intelligence</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-blue-400 mb-2">
                        {Math.round(ultraAnalysis.profile_insights.emotional_intelligence * 100)}%
                      </div>
                      <Progress value={ultraAnalysis.profile_insights.emotional_intelligence * 100} />
                    </CardContent>
                  </Card>

                  <Card className="border-gray-700 bg-gray-800">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Creativity Index</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-purple-400 mb-2">
                        {Math.round(ultraAnalysis.profile_insights.creativity_index * 100)}%
                      </div>
                      <Progress value={ultraAnalysis.profile_insights.creativity_index * 100} />
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="jobs" className="space-y-4">
                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Rocket className="h-5 w-5 text-green-500" />
                      AI-Powered Job Recommendations
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                      Jobs analyzed with advanced AI algorithms and predictive modeling
                    </CardDescription>
                  </CardHeader>
                </Card>

                {ultraAnalysis.ai_powered_jobs.map((job, index) => (
                  <Card key={index} className="border-l-4 border-l-blue-500 border-gray-700 bg-gray-800">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">{job.title}</CardTitle>
                      <CardDescription className="text-gray-300">{job.company} • {job.location}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm text-gray-300">{job.description}</p>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-400">
                            {Math.round(job.success_probability * 100)}%
                          </div>
                          <div className="text-xs text-gray-500">Success Probability</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-400">
                            {Math.round(job.performance_prediction * 100)}%
                          </div>
                          <div className="text-xs text-gray-500">Performance</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-400">
                            {Math.round(job.retention_likelihood * 100)}%
                          </div>
                          <div className="text-xs text-gray-500">Retention</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-orange-400">
                            {Math.round(job.career_impact * 100)}%
                          </div>
                          <div className="text-xs text-gray-500">Career Impact</div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-gray-500" />
                          <span className="text-sm text-gray-300">Promotion Timeline: {job.promotion_timeline}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <AlertCircle className="h-4 w-4 text-yellow-500" />
                          <span className="text-sm text-gray-300">Automation Risk: {Math.round(job.automation_risk * 100)}%</span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div>
                          <h5 className="font-semibold text-sm mb-1 text-white">Matching Skills:</h5>
                          <div className="flex flex-wrap gap-1">
                            {job.matching_skills.map((skill: string, i: number) => (
                              <Badge key={i} variant="secondary" className="text-xs bg-green-600 text-white">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h5 className="font-semibold text-sm mb-1 text-white">Skills to Develop:</h5>
                          <div className="flex flex-wrap gap-1">
                            {job.missing_skills.map((skill: string, i: number) => (
                              <Badge key={i} variant="outline" className="text-xs text-gray-400 border-gray-600">
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              <TabsContent value="insights" className="space-y-4">
                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-white">
                      <Award className="h-5 w-5 text-yellow-500" />
                      AI System Information
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h4 className="font-semibold mb-2 text-white">Analysis Depth</h4>
                      <Badge className="bg-purple-600 text-white">
                        {ultraAnalysis.ai_system_info.analysis_depth}
                      </Badge>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold mb-2 text-white">AI Models Used</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {ultraAnalysis.ai_system_info.ai_models_used.map((model, index) => (
                          <Badge key={index} variant="outline" className="justify-start text-gray-300 border-gray-600">
                            <Cpu className="h-3 w-3 mr-1" />
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2 text-white">Features Enabled</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {ultraAnalysis.ai_system_info.features_enabled.map((feature, index) => (
                          <Badge key={index} variant="secondary" className="justify-start bg-blue-600 text-white">
                            <Sparkles className="h-3 w-3 mr-1" />
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-gray-700 bg-gray-800">
                  <CardHeader>
                    <CardTitle className="text-white">Career Aspirations</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {ultraAnalysis.profile_insights.career_aspirations.map((aspiration, index) => (
                        <Badge key={index} className="bg-blue-600 text-white">
                          <Target className="h-3 w-3 mr-1" />
                          {aspiration}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-4">
          <Card className="border-gray-700 bg-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Personalized Job Recommendations</CardTitle>
              <CardDescription className="text-gray-400">
                AI-powered job matches based on your resume analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={getJobRecommendations} 
                disabled={loading || profileLoading}
                className="bg-blue-600 hover:bg-blue-700 text-white"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Finding Jobs...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Get Recommendations
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {jobRecommendations.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">
                Found {jobRecommendations.length} Job Recommendations
              </h3>
              {jobRecommendations.map((job, index) => (
                <JobCard key={index} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Search Tab */}
        <TabsContent value="search" className="space-y-4">
          <Card className="border-gray-700 bg-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Advanced Job Search</CardTitle>
              <CardDescription className="text-gray-400">
                Search for jobs with custom filters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Keywords (comma-separated)
                  </label>
                  <Input
                    placeholder="e.g., React, JavaScript, Frontend"
                    value={searchKeywords}
                    onChange={(e) => setSearchKeywords(e.target.value)}
                    className="bg-gray-700 border-gray-600 text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Location
                  </label>
                  <Input
                    placeholder="e.g., Remote, New York, London"
                    value={searchLocation}
                    onChange={(e) => setSearchLocation(e.target.value)}
                    className="bg-gray-700 border-gray-600 text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Job Type
                  </label>
                  <select
                    value={jobType}
                    onChange={(e) => setJobType(e.target.value)}
                    className="w-full h-10 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  >
                    <option value="">Any</option>
                    <option value="full-time">Full-time</option>
                    <option value="part-time">Part-time</option>
                    <option value="contract">Contract</option>
                    <option value="remote">Remote</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-white mb-2">
                    Experience Level
                  </label>
                  <select
                    value={experienceLevel}
                    onChange={(e) => setExperienceLevel(e.target.value)}
                    className="w-full h-10 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                  >
                    <option value="">Any</option>
                    <option value="entry">Entry Level</option>
                    <option value="mid">Mid Level</option>
                    <option value="senior">Senior Level</option>
                    <option value="lead">Lead/Principal</option>
                  </select>
                </div>
              </div>
              <Button 
                onClick={searchJobs} 
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 mr-2" />
                    Search Jobs
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {searchResults.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">
                Found {searchResults.length} Jobs
              </h3>
              {searchResults.map((job, index) => (
                <JobCard key={index} job={job} />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Saved Jobs Tab */}
        <TabsContent value="saved" className="space-y-4">
          <Card className="border-gray-700 bg-gray-800">
            <CardHeader>
              <CardTitle className="text-white">Your Saved Jobs</CardTitle>
              <CardDescription className="text-gray-400">
                Jobs you've saved for later review
              </CardDescription>
            </CardHeader>
          </Card>

          {savedJobs.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">
                {savedJobs.length} Saved Jobs
              </h3>
              {savedJobs.map((job, index) => (
                <JobCard 
                  key={index} 
                  job={job} 
                  showSaveButton={false}
                  showRemoveButton={true}
                />
              ))}
            </div>
          ) : (
            <Card className="border-gray-700 bg-gray-800">
              <CardContent className="p-8 text-center">
                <Heart className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">No Saved Jobs</h3>
                <p className="text-gray-400">Start exploring job recommendations and save the ones you like!</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}