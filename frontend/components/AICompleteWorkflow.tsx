'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Scan, RefreshCw, Download, Briefcase, CheckCircle, Clock, Zap } from 'lucide-react';
import { Button } from '../app/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../app/components/ui/card';
import { Progress } from '../app/components/ui/progress';
import { Badge } from '../app/components/ui/badge';
import { Alert, AlertDescription } from '../app/components/ui/alert';
import axios from 'axios';

interface WorkflowStep {
  id: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  status: 'pending' | 'processing' | 'completed' | 'error';
}

interface JobSuggestion {
  title: string;
  company: string;
  location: string;
  description: string;
  match_score: number;
  ai_insights: {
    skill_match: number;
    experience_fit: number;
    location_preference: number;
  };
}

interface WorkflowResponse {
  status: string;
  workflow_complete: boolean;
  resume_id: number;
  step_1_upload: {
    status: string;
    file_processed: string;
    text_extracted: boolean;
  };
  step_2_scan: {
    status: string;
    original_ats_score: number;
    scan_results: any;
    ats_friendly: boolean;
  };
  step_3_refine: {
    status: string;
    improvements_applied: boolean;
    final_ats_score: number;
    score_improvement: number;
    improvements_made: string[];
  };
  step_4_download: {
    status: string;
    improved_resume_text: string;
    download_filename: string;
  };
  step_5_jobs: {
    status: string;
    job_suggestions: JobSuggestion[];
    total_jobs_found: number;
    location_searched: string;
  };
  ai_summary: {
    total_processing_time: string;
    ai_models_used: string[];
    confidence_score: number;
    recommendation: string;
  };
}

const AICompleteWorkflow: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [workflowResult, setWorkflowResult] = useState<WorkflowResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [userLocation, setUserLocation] = useState('Remote');

  const [steps, setSteps] = useState<WorkflowStep[]>([
    {
      id: 1,
      title: 'Upload Resume',
      description: 'AI processes your resume file',
      icon: <Upload className="w-5 h-5" />,
      status: 'pending'
    },
    {
      id: 2,
      title: 'ATS Scan',
      description: 'AI analyzes ATS friendliness',
      icon: <Scan className="w-5 h-5" />,
      status: 'pending'
    },
    {
      id: 3,
      title: 'AI Refinement',
      description: 'AI optimizes your resume',
      icon: <RefreshCw className="w-5 h-5" />,
      status: 'pending'
    },
    {
      id: 4,
      title: 'Download Ready',
      description: 'Improved resume available',
      icon: <Download className="w-5 h-5" />,
      status: 'pending'
    },
    {
      id: 5,
      title: 'Job Suggestions',
      description: 'AI finds matching opportunities',
      icon: <Briefcase className="w-5 h-5" />,
      status: 'pending'
    }
  ]);

  const updateStepStatus = (stepId: number, status: WorkflowStep['status']) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, status } : step
    ));
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    setWorkflowResult(null);
    setCurrentStep(1);

    // Reset all steps
    setSteps(prev => prev.map(step => ({ ...step, status: 'pending' })));

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('location', userLocation);

      // Get auth token if available
      const token = localStorage.getItem('token');
      const headers: any = {
        'Content-Type': 'multipart/form-data',
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      // Step 1: Start processing
      updateStepStatus(1, 'processing');
      
      // Simulate step progression for better UX
      setTimeout(() => {
        setCurrentStep(2);
        updateStepStatus(1, 'completed');
        updateStepStatus(2, 'processing');
      }, 1000);

      setTimeout(() => {
        setCurrentStep(3);
        updateStepStatus(2, 'completed');
        updateStepStatus(3, 'processing');
      }, 2000);

      setTimeout(() => {
        setCurrentStep(4);
        updateStepStatus(3, 'completed');
        updateStepStatus(4, 'processing');
      }, 3000);

      setTimeout(() => {
        setCurrentStep(5);
        updateStepStatus(4, 'completed');
        updateStepStatus(5, 'processing');
      }, 4000);

      const response = await axios.post(
        'http://localhost:5000/api/ai-workflow/complete',
        formData,
        { headers }
      );

      if (response.data.status === 'success') {
        setWorkflowResult(response.data);
        
        // Mark all steps as completed
        setSteps(prev => prev.map(step => ({ ...step, status: 'completed' })));
        setCurrentStep(6);
      } else {
        throw new Error(response.data.error || 'Workflow failed');
      }

    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'An error occurred during processing');
      
      // Mark current step as error
      if (currentStep > 0) {
        updateStepStatus(currentStep, 'error');
      }
    } finally {
      setIsProcessing(false);
    }
  }, [userLocation, currentStep]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
    disabled: isProcessing
  });

  const downloadImprovedResume = () => {
    if (!workflowResult?.step_4_download.improved_resume_text) return;

    const blob = new Blob([workflowResult.step_4_download.improved_resume_text], {
      type: 'text/plain'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = workflowResult.step_4_download.download_filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getStepIcon = (step: WorkflowStep) => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'error':
        return <div className="w-5 h-5 rounded-full bg-red-500" />;
      default:
        return step.icon;
    }
  };

  const progressPercentage = (currentStep / 5) * 100;

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-2">
          <Zap className="w-8 h-8 text-blue-500" />
          <h1 className="text-3xl font-bold">AI Complete Workflow</h1>
        </div>
        <p className="text-gray-600">
          Upload your resume and let AI handle everything: scan, optimize, and find jobs for you!
        </p>
      </div>

      {/* Location Input */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Job Search Location</CardTitle>
          <CardDescription>
            Enter your preferred location for job suggestions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <input
            type="text"
            value={userLocation}
            onChange={(e) => setUserLocation(e.target.value)}
            placeholder="e.g., New York, Remote, San Francisco"
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isProcessing}
          />
        </CardContent>
      </Card>

      {/* File Upload */}
      {!workflowResult && (
        <Card>
          <CardHeader>
            <CardTitle>Upload Your Resume</CardTitle>
            <CardDescription>
              Drag and drop your resume file or click to browse (PDF, DOCX, TXT - Max 10MB)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-blue-500 bg-blue-50'
                  : isProcessing
                  ? 'border-gray-300 bg-gray-50 cursor-not-allowed'
                  : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'
              }`}
            >
              <input {...getInputProps()} />
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              {isProcessing ? (
                <p className="text-gray-500">Processing your resume...</p>
              ) : isDragActive ? (
                <p className="text-blue-600">Drop your resume here...</p>
              ) : (
                <div>
                  <p className="text-gray-600 mb-2">
                    Drag and drop your resume here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PDF, DOCX, and TXT files up to 10MB
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Progress Steps */}
      {isProcessing && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-blue-500" />
              AI Processing Your Resume
            </CardTitle>
            <CardDescription>
              Please wait while our AI handles all the steps automatically
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={progressPercentage} className="w-full" />
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {steps.map((step) => (
                <div
                  key={step.id}
                  className={`p-4 rounded-lg border ${
                    step.status === 'completed'
                      ? 'bg-green-50 border-green-200'
                      : step.status === 'processing'
                      ? 'bg-blue-50 border-blue-200'
                      : step.status === 'error'
                      ? 'bg-red-50 border-red-200'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {getStepIcon(step)}
                    <span className="font-medium text-sm">{step.title}</span>
                  </div>
                  <p className="text-xs text-gray-600">{step.description}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription className="text-red-700">
            {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Results */}
      {workflowResult && (
        <div className="space-y-6">
          {/* AI Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-green-600">
                <CheckCircle className="w-6 h-6" />
                AI Workflow Complete!
              </CardTitle>
              <CardDescription>
                {workflowResult.ai_summary.recommendation}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {workflowResult.step_2_scan.original_ats_score}→{workflowResult.step_3_refine.final_ats_score}
                  </div>
                  <div className="text-sm text-gray-600">ATS Score Improvement</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {workflowResult.ai_summary.confidence_score}%
                  </div>
                  <div className="text-sm text-gray-600">AI Confidence</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {workflowResult.step_5_jobs.total_jobs_found}
                  </div>
                  <div className="text-sm text-gray-600">Job Matches Found</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Download Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="w-5 h-5" />
                Download Your Optimized Resume
              </CardTitle>
              <CardDescription>
                Your resume has been improved with AI optimizations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {workflowResult.step_3_refine.improvements_made.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">AI Improvements Applied:</h4>
                    <div className="flex flex-wrap gap-2">
                      {workflowResult.step_3_refine.improvements_made.map((improvement, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {improvement}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                <Button onClick={downloadImprovedResume} className="w-full">
                  <Download className="w-4 h-4 mr-2" />
                  Download Optimized Resume
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Job Suggestions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Briefcase className="w-5 h-5" />
                AI Job Suggestions
              </CardTitle>
              <CardDescription>
                Jobs matched to your optimized resume in {workflowResult.step_5_jobs.location_searched}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {workflowResult.step_5_jobs.job_suggestions.map((job, index) => (
                  <div key={index} className="border rounded-lg p-4 space-y-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-semibold text-lg">{job.title}</h4>
                        <p className="text-gray-600">{job.company} • {job.location}</p>
                      </div>
                      <Badge variant="outline" className="text-green-600 border-green-600">
                        {job.match_score}% Match
                      </Badge>
                    </div>
                    <p className="text-gray-700 text-sm">{job.description}</p>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Skill Match:</span>
                        <div className="font-medium">{job.ai_insights.skill_match}%</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Experience Fit:</span>
                        <div className="font-medium">{job.ai_insights.experience_fit}%</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Location:</span>
                        <div className="font-medium">{job.ai_insights.location_preference}%</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Start Over Button */}
          <div className="text-center">
            <Button
              variant="outline"
              onClick={() => {
                setWorkflowResult(null);
                setError(null);
                setCurrentStep(0);
                setSteps(prev => prev.map(step => ({ ...step, status: 'pending' })));
              }}
            >
              Process Another Resume
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AICompleteWorkflow;