'use client'

import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Target, 
  Lightbulb, 
  RotateCcw,
  TrendingUp,
  Award,
  FileText
} from 'lucide-react'

interface ResultsDisplayProps {
  results: any
  onStartOver: () => void
}

export default function ResultsDisplay({ results, onStartOver }: ResultsDisplayProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-success-600'
    if (score >= 60) return 'text-warning-600'
    return 'text-danger-600'
  }

  const getScoreBgColor = (score: number) => {
    if (score >= 80) return 'bg-success-100'
    if (score >= 60) return 'bg-warning-100'
    return 'bg-danger-100'
  }

  const getScoreIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="h-8 w-8 text-success-600" />
    if (score >= 60) return <AlertTriangle className="h-8 w-8 text-warning-600" />
    return <XCircle className="h-8 w-8 text-danger-600" />
  }

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-800'
      case 'medium': return 'bg-yellow-100 text-yellow-800'
      case 'low': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Overall Score Card */}
      <div className="card text-center">
        <div className="flex flex-col items-center space-y-4">
          <div className={`p-4 rounded-full ${getScoreBgColor(results.ats_score)}`}>
            {getScoreIcon(results.ats_score)}
          </div>
          
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              ATS Score: <span className={getScoreColor(results.ats_score)}>
                {results.ats_score}%
              </span>
            </h2>
            <p className="text-lg text-gray-600 mb-4">
              Overall Rating: <span className="font-semibold">{results.overall_rating}</span>
            </p>
          </div>

          <div className="w-full max-w-md">
            <div className="bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-1000 ${
                  results.ats_score >= 80 ? 'bg-success-500' :
                  results.ats_score >= 60 ? 'bg-warning-500' : 'bg-danger-500'
                }`}
                style={{ width: `${results.ats_score}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Keyword Analysis */}
      {results.keyword_analysis && results.keyword_analysis.match_percentage > 0 && (
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-primary-100 p-2 rounded-lg">
              <Target className="h-5 w-5 text-primary-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Keyword Analysis</h3>
              <p className="text-sm text-gray-600">
                Match Rate: {results.keyword_analysis.match_percentage}%
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Matched Keywords */}
            {results.keyword_analysis.matched.length > 0 && (
              <div>
                <h4 className="font-medium text-success-700 mb-3 flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4" />
                  <span>Matched Keywords ({results.keyword_analysis.matched.length})</span>
                </h4>
                <div className="flex flex-wrap gap-2">
                  {results.keyword_analysis.matched.slice(0, 10).map((keyword: string, index: number) => (
                    <span 
                      key={index}
                      className="px-2 py-1 bg-success-100 text-success-800 text-xs rounded-full"
                    >
                      {keyword}
                    </span>
                  ))}
                  {results.keyword_analysis.matched.length > 10 && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                      +{results.keyword_analysis.matched.length - 10} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Missing Keywords */}
            {results.keyword_analysis.missing.length > 0 && (
              <div>
                <h4 className="font-medium text-danger-700 mb-3 flex items-center space-x-2">
                  <XCircle className="h-4 w-4" />
                  <span>Missing Keywords ({results.keyword_analysis.missing.length})</span>
                </h4>
                <div className="flex flex-wrap gap-2">
                  {results.keyword_analysis.missing.slice(0, 10).map((keyword: string, index: number) => (
                    <span 
                      key={index}
                      className="px-2 py-1 bg-danger-100 text-danger-800 text-xs rounded-full"
                    >
                      {keyword}
                    </span>
                  ))}
                  {results.keyword_analysis.missing.length > 10 && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                      +{results.keyword_analysis.missing.length - 10} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* AI Suggestions */}
      {results.suggestions && results.suggestions.length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="bg-warning-100 p-2 rounded-lg">
              <Lightbulb className="h-5 w-5 text-warning-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">AI-Powered Suggestions</h3>
              <p className="text-sm text-gray-600">
                Personalized recommendations to improve your resume
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {results.suggestions.map((suggestion: any, index: number) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-900">{suggestion.category}</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${getPriorityColor(suggestion.priority)}`}>
                      {suggestion.priority}
                    </span>
                  </div>
                </div>
                <p className="text-gray-700 text-sm">{suggestion.suggestion}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={onStartOver}
          className="btn-primary flex items-center justify-center space-x-2"
        >
          <RotateCcw className="h-4 w-4" />
          <span>Analyze Another Resume</span>
        </button>
        
        <button
          onClick={() => window.print()}
          className="btn-secondary flex items-center justify-center space-x-2"
        >
          <FileText className="h-4 w-4" />
          <span>Save Results</span>
        </button>
      </div>

      {/* Tips for Improvement */}
      <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <div className="flex items-center space-x-3 mb-4">
          <div className="bg-blue-100 p-2 rounded-lg">
            <TrendingUp className="h-5 w-5 text-blue-600" />
          </div>
          <div>
            <h3 className="font-semibold text-blue-900">Next Steps</h3>
            <p className="text-sm text-blue-700">How to improve your resume further</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-start space-x-3">
            <Award className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-blue-900">Optimize for ATS</h4>
              <p className="text-sm text-blue-700">
                Use standard section headings, avoid tables and images, and include relevant keywords.
              </p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <Target className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-blue-900">Tailor for Each Job</h4>
              <p className="text-sm text-blue-700">
                Customize your resume for each application by including job-specific keywords.
              </p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-blue-900">Regular Updates</h4>
              <p className="text-sm text-blue-700">
                Keep your resume updated with new skills, experiences, and achievements.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}