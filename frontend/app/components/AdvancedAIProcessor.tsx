'use client';

import React, { useState, useRef } from 'react';
import { Upload, Brain, Zap, Download, Star, TrendingUp, Settings, FileText, Eye, Sparkles } from 'lucide-react';

interface AnalysisResult {
  content_analysis: {
    sections: any;
    quality_scores: any;
    missing_sections: string[];
    recommendations: string[];
  };
  formatting_analysis: {
    font_info: any;
    layout_structure: any;
    color_scheme: any;
    spacing_metrics: any;
  };
  structure_analysis: {
    section_hierarchy: any;
    content_flow: any;
    readability_score: number;
  };
  ats_compatibility: {
    ats_score: number;
    compatibility_issues: string[];
    optimization_suggestions: string[];
  };
  improvement_suggestions: {
    content_improvements: string[];
    formatting_improvements: string[];
    keyword_suggestions: string[];
    structure_improvements: string[];
  };
}

interface ImprovedResume {
  improved_content: any;
  formatting_preserved: boolean;
  improvements_applied: string[];
  quality_improvement: number;
  ats_score_improvement: number;
}

const AdvancedAIProcessor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isImproving, setIsImproving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [improvedResume, setImprovedResume] = useState<ImprovedResume | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [improvementPreferences, setImprovementPreferences] = useState({
    focus_areas: ['content', 'formatting', 'ats_optimization'],
    aggressiveness: 'moderate',
    preserve_style: true,
    target_role: '',
    industry: ''
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setAnalysisResult(null);
      setImprovedResume(null);
    }
  };

  const performDeepAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/api/ai/deep-analyze', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      });

      const data = await response.json();
      if (data.status === 'success') {
        setAnalysisResult(data.analysis);
        setActiveTab('analysis');
      } else {
        alert('Analysis failed: ' + data.error);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const performIntelligentImprovement = async () => {
    if (!analysisResult) return;

    setIsImproving(true);
    try {
      const response = await fetch('/api/ai/intelligent-improve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          analysis_result: analysisResult,
          preferences: improvementPreferences
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        setImprovedResume(data.improved_resume);
        setActiveTab('improvements');
      } else {
        alert('Improvement failed: ' + data.error);
      }
    } catch (error) {
      console.error('Improvement error:', error);
      alert('Improvement failed. Please try again.');
    } finally {
      setIsImproving(false);
    }
  };

  const generateFormattedResume = async (format: string = 'pdf') => {
    if (!improvedResume || !analysisResult) return;

    setIsGenerating(true);
    try {
      const response = await fetch('/api/ai/generate-formatted-resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          improved_content: improvedResume.improved_content,
          original_formatting: analysisResult.formatting_analysis,
          output_format: format
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        // Convert base64 to blob and download
        const byteCharacters = atob(data.resume_data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: format === 'pdf' ? 'application/pdf' : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `improved_resume.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Generation failed: ' + data.error);
      }
    } catch (error) {
      console.error('Generation error:', error);
      alert('Generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const getTrainingStatus = async () => {
    try {
      const response = await fetch('/api/ai/advanced-training-status', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      const data = await response.json();
      if (data.status === 'success') {
        setTrainingStatus(data.training_status);
      }
    } catch (error) {
      console.error('Training status error:', error);
    }
  };

  const triggerAILearning = async (learningType: string) => {
    try {
      const response = await fetch('/api/ai/advanced-trigger-learning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ learning_type: learningType })
      });

      const data = await response.json();
      if (data.status === 'success') {
        alert(data.message);
        getTrainingStatus(); // Refresh status
      } else {
        alert('Learning trigger failed: ' + data.error);
      }
    } catch (error) {
      console.error('Learning trigger error:', error);
      alert('Learning trigger failed. Please try again.');
    }
  };

  const submitFeedback = async (feedbackType: string, rating: number, comments: string = '') => {
    try {
      const response = await fetch('/api/ai/advanced-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          feedback_type: feedbackType,
          rating,
          comments,
          resume_id: selectedFile?.name
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        alert('Feedback submitted successfully!');
      } else {
        alert('Feedback submission failed: ' + data.error);
      }
    } catch (error) {
      console.error('Feedback error:', error);
      alert('Feedback submission failed. Please try again.');
    }
  };

  React.useEffect(() => {
    getTrainingStatus();
  }, []);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-gray-800 mb-2 flex items-center">
          <Brain className="mr-3 text-blue-600" />
          Advanced AI Resume Processor
        </h2>
        <p className="text-gray-600">
          Intelligent resume analysis with formatting preservation and self-learning capabilities
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
        {[
          { id: 'upload', label: 'Upload & Analyze', icon: Upload },
          { id: 'analysis', label: 'Deep Analysis', icon: Eye },
          { id: 'improvements', label: 'AI Improvements', icon: Sparkles },
          { id: 'training', label: 'AI Training', icon: TrendingUp },
          { id: 'settings', label: 'Preferences', icon: Settings }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <tab.icon className="w-4 h-4 mr-2" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Upload & Analyze Tab */}
      {activeTab === 'upload' && (
        <div className="space-y-6">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.doc,.docx"
              onChange={handleFileSelect}
              className="hidden"
            />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Upload Your Resume
            </h3>
            <p className="text-gray-500 mb-4">
              Supports PDF, DOC, and DOCX formats
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Choose File
            </button>
            {selectedFile && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  Selected: <span className="font-medium">{selectedFile.name}</span>
                </p>
                <p className="text-xs text-gray-500">
                  Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="flex justify-center">
              <button
                onClick={performDeepAnalysis}
                disabled={isAnalyzing}
                className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2" />
                    Start Deep Analysis
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Analysis Results Tab */}
      {activeTab === 'analysis' && analysisResult && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-800 mb-2">Content Quality</h4>
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(analysisResult.content_analysis.quality_scores?.overall || 0)}%
              </div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">ATS Score</h4>
              <div className="text-2xl font-bold text-green-600">
                {Math.round(analysisResult.ats_compatibility.ats_score || 0)}%
              </div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-800 mb-2">Readability</h4>
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(analysisResult.structure_analysis.readability_score || 0)}%
              </div>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <h4 className="font-semibold text-orange-800 mb-2">Sections Found</h4>
              <div className="text-2xl font-bold text-orange-600">
                {Object.keys(analysisResult.content_analysis.sections || {}).length}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-6 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
                <FileText className="mr-2" />
                Content Recommendations
              </h4>
              <ul className="space-y-2">
                {analysisResult.content_analysis.recommendations?.map((rec, index) => (
                  <li key={index} className="text-sm text-gray-700 flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>

            <div className="bg-gray-50 p-6 rounded-lg">
              <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
                <Zap className="mr-2" />
                ATS Optimization
              </h4>
              <ul className="space-y-2">
                {analysisResult.ats_compatibility.optimization_suggestions?.map((suggestion, index) => (
                  <li key={index} className="text-sm text-gray-700 flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={performIntelligentImprovement}
              disabled={isImproving}
              className="bg-purple-600 text-white px-8 py-3 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center"
            >
              {isImproving ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Improving...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2" />
                  Apply AI Improvements
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Improvements Tab */}
      {activeTab === 'improvements' && improvedResume && (
        <div className="space-y-6">
          <div className="bg-green-50 p-6 rounded-lg">
            <h4 className="font-semibold text-green-800 mb-4">Improvement Summary</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-green-600">Quality Improvement</p>
                <p className="text-2xl font-bold text-green-800">
                  +{Math.round(improvedResume.quality_improvement || 0)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-green-600">ATS Score Improvement</p>
                <p className="text-2xl font-bold text-green-800">
                  +{Math.round(improvedResume.ats_score_improvement || 0)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-green-600">Formatting Preserved</p>
                <p className="text-2xl font-bold text-green-800">
                  {improvedResume.formatting_preserved ? '✓' : '✗'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-4">Applied Improvements</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {improvedResume.improvements_applied?.map((improvement, index) => (
                <div key={index} className="flex items-center text-sm text-gray-700">
                  <span className="text-green-500 mr-2">✓</span>
                  {improvement}
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-center space-x-4">
            <button
              onClick={() => generateFormattedResume('pdf')}
              disabled={isGenerating}
              className="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 flex items-center"
            >
              {isGenerating ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              ) : (
                <Download className="mr-2" />
              )}
              Download PDF
            </button>
            <button
              onClick={() => generateFormattedResume('docx')}
              disabled={isGenerating}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
            >
              {isGenerating ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              ) : (
                <Download className="mr-2" />
              )}
              Download DOCX
            </button>
          </div>

          {/* Feedback Section */}
          <div className="bg-yellow-50 p-6 rounded-lg">
            <h4 className="font-semibold text-yellow-800 mb-4">Rate the AI Improvements</h4>
            <div className="flex items-center space-x-4">
              {[1, 2, 3, 4, 5].map(rating => (
                <button
                  key={rating}
                  onClick={() => submitFeedback('satisfaction', rating)}
                  className="text-yellow-400 hover:text-yellow-500 text-2xl"
                >
                  <Star className="w-8 h-8" />
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Training Status Tab */}
      {activeTab === 'training' && (
        <div className="space-y-6">
          {trainingStatus && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 p-6 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Model Accuracy</h4>
                <div className="text-3xl font-bold text-blue-600">
                  {Math.round((trainingStatus.model_performance?.accuracy || 0) * 100)}%
                </div>
              </div>
              <div className="bg-green-50 p-6 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Training Samples</h4>
                <div className="text-3xl font-bold text-green-600">
                  {trainingStatus.training_data_size || 0}
                </div>
              </div>
              <div className="bg-purple-50 p-6 rounded-lg">
                <h4 className="font-semibold text-purple-800 mb-2">Last Training</h4>
                <div className="text-sm text-purple-600">
                  {trainingStatus.last_training_time || 'Never'}
                </div>
              </div>
            </div>
          )}

          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-4">AI Learning Controls</h4>
            <div className="flex space-x-4">
              <button
                onClick={() => triggerAILearning('incremental')}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Incremental Learning
              </button>
              <button
                onClick={() => triggerAILearning('full')}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
              >
                Full Retraining
              </button>
              <button
                onClick={() => triggerAILearning('external')}
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
              >
                External Data Learning
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === 'settings' && (
        <div className="space-y-6">
          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-800 mb-4">Improvement Preferences</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Focus Areas
                </label>
                <div className="space-y-2">
                  {['content', 'formatting', 'ats_optimization', 'keywords'].map(area => (
                    <label key={area} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={improvementPreferences.focus_areas.includes(area)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setImprovementPreferences(prev => ({
                              ...prev,
                              focus_areas: [...prev.focus_areas, area]
                            }));
                          } else {
                            setImprovementPreferences(prev => ({
                              ...prev,
                              focus_areas: prev.focus_areas.filter(a => a !== area)
                            }));
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="text-sm text-gray-700 capitalize">
                        {area.replace('_', ' ')}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Improvement Aggressiveness
                </label>
                <select
                  value={improvementPreferences.aggressiveness}
                  onChange={(e) => setImprovementPreferences(prev => ({
                    ...prev,
                    aggressiveness: e.target.value
                  }))}
                  className="w-full p-2 border border-gray-300 rounded-lg"
                >
                  <option value="conservative">Conservative</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedAIProcessor;