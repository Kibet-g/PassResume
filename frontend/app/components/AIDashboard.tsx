'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, Brain, TrendingUp, Database, Globe, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';

interface TrainingStats {
  total_resumes_analyzed: number;
  successful_improvements: number;
  average_improvement_score: number;
  model_accuracy: number;
  last_external_learning: string;
  patterns_learned: number;
}

interface AIDashboardProps {
  userRole?: string;
}

export default function AIDashboard({ userRole = 'user' }: AIDashboardProps) {
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [lastTraining, setLastTraining] = useState<string>('');
  const [modelVersion, setModelVersion] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTriggering, setIsTriggering] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  const fetchTrainingStatus = async () => {
    setIsLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${apiUrl}/api/ai-training-status`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch training status');
      }

      const data = await response.json();
      setTrainingStats(data.training_stats);
      setLastTraining(data.last_training);
      setModelVersion(data.model_version);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load training status');
    } finally {
      setIsLoading(false);
    }
  };

  const triggerAILearning = async (includeExternal: boolean = false) => {
    setIsTriggering(true);
    setError('');
    setSuccess('');

    try {
      const token = localStorage.getItem('token');
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${apiUrl}/api/trigger-ai-learning`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ include_external: includeExternal })
      });

      if (!response.ok) {
        throw new Error('Failed to trigger AI learning');
      }

      setSuccess('AI learning process triggered successfully');
      // Refresh status after a delay
      setTimeout(() => {
        fetchTrainingStatus();
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger learning');
    } finally {
      setIsTriggering(false);
    }
  };

  useEffect(() => {
    fetchTrainingStatus();
  }, []);

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-green-600';
    if (accuracy >= 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAccuracyBadgeVariant = (accuracy: number) => {
    if (accuracy >= 90) return 'default';
    if (accuracy >= 80) return 'secondary';
    return 'destructive';
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          AI Learning Dashboard
        </CardTitle>
        <CardDescription>
          Monitor AI training progress and trigger learning processes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2">
          <Button 
            onClick={() => fetchTrainingStatus()} 
            disabled={isLoading}
            variant="outline"
            size="sm"
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh Status
          </Button>
        </div>

        {trainingStats && (
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="controls">Controls</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Model Version</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{modelVersion || 'v1.0.0'}</div>
                    <p className="text-sm text-gray-600">Current AI model version</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Last Training</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-sm font-medium">{formatDate(lastTraining)}</div>
                    <p className="text-sm text-gray-600">Most recent learning session</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Resumes Analyzed</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{trainingStats.total_resumes_analyzed.toLocaleString()}</div>
                    <p className="text-sm text-gray-600">Total resumes processed</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Patterns Learned</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{trainingStats.patterns_learned.toLocaleString()}</div>
                    <p className="text-sm text-gray-600">Unique patterns identified</p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="performance" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Model Accuracy</span>
                    <Badge variant={getAccuracyBadgeVariant(trainingStats.model_accuracy)}>
                      {trainingStats.model_accuracy.toFixed(1)}%
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${
                        trainingStats.model_accuracy >= 90 ? 'bg-green-500' : 
                        trainingStats.model_accuracy >= 80 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${trainingStats.model_accuracy}%` }}
                    ></div>
                  </div>
                  <p className={`text-sm mt-2 ${getAccuracyColor(trainingStats.model_accuracy)}`}>
                    {trainingStats.model_accuracy >= 90 ? 'Excellent model performance' :
                     trainingStats.model_accuracy >= 80 ? 'Good model performance' :
                     'Model needs improvement'}
                  </p>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Successful Improvements
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{trainingStats.successful_improvements.toLocaleString()}</div>
                    <p className="text-sm text-gray-600">Resumes successfully enhanced</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Average Improvement</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{trainingStats.average_improvement_score.toFixed(1)}%</div>
                    <p className="text-sm text-gray-600">Average score improvement</p>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">External Learning</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-sm font-medium">{formatDate(trainingStats.last_external_learning)}</div>
                  <p className="text-sm text-gray-600">Last external data integration</p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="controls" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Database Learning
                  </CardTitle>
                  <CardDescription>
                    Train the AI using anonymized data from the application database
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={() => triggerAILearning(false)} 
                    disabled={isTriggering}
                    className="w-full"
                  >
                    {isTriggering ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Triggering Learning...
                      </>
                    ) : (
                      <>
                        <Database className="mr-2 h-4 w-4" />
                        Trigger Database Learning
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Globe className="h-5 w-5" />
                    External Data Learning
                  </CardTitle>
                  <CardDescription>
                    Enhance AI with publicly available resume patterns and best practices
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={() => triggerAILearning(true)} 
                    disabled={isTriggering}
                    variant="outline"
                    className="w-full"
                  >
                    {isTriggering ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Triggering Learning...
                      </>
                    ) : (
                      <>
                        <Globe className="mr-2 h-4 w-4" />
                        Trigger External Learning
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Learning processes may take several minutes to complete. The AI will continue to improve automatically based on user interactions.
                </AlertDescription>
              </Alert>
            </TabsContent>
          </Tabs>
        )}

        {!trainingStats && !isLoading && (
          <Card>
            <CardContent className="text-center py-8">
              <p className="text-gray-500">No training data available. Click refresh to load status.</p>
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  );
}