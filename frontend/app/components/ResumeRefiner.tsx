'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, Wand2, CheckCircle, AlertCircle, Download, Copy } from 'lucide-react';

interface Improvement {
  type: string;
  original: string;
  improved: string;
  reason: string;
  impact: string;
}

interface ResumeRefinerProps {
  resumeText: string;
  jobDescription?: string;
  onRefinementComplete: (refinedResume: string, improvements: Improvement[]) => void;
}

export default function ResumeRefiner({ resumeText, jobDescription = '', onRefinementComplete }: ResumeRefinerProps) {
  const [isRefining, setIsRefining] = useState(false);
  const [refinedResume, setRefinedResume] = useState<string>('');
  const [improvements, setImprovements] = useState<Improvement[]>([]);
  const [error, setError] = useState<string>('');
  const [copied, setCopied] = useState(false);

  const performRefinement = async () => {
    if (!resumeText.trim()) {
      setError('Please provide resume text to refine');
      return;
    }

    setIsRefining(true);
    setError('');

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${apiUrl}/api/refine-resume-public`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          resume_text: resumeText,
          job_description: jobDescription 
        })
      });

      if (!response.ok) {
        throw new Error('Failed to refine resume');
      }

      const data = await response.json();
      setRefinedResume(data.refined_resume);
      setImprovements(data.improvements);
      onRefinementComplete(data.refined_resume, data.improvements);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Refinement failed');
    } finally {
      setIsRefining(false);
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(refinedResume);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'low': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const getImpactBadgeVariant = (impact: string) => {
    switch (impact.toLowerCase()) {
      case 'high': return 'default';
      case 'medium': return 'secondary';
      case 'low': return 'outline';
      default: return 'outline';
    }
  };

  return (
    <Card className="w-full bg-gray-800/50 border-gray-600">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <Wand2 className="h-5 w-5 text-purple-400" />
          AI Resume Refiner
        </CardTitle>
        <CardDescription className="text-gray-400">
          Intelligent resume enhancement with AI-powered improvements and optimization
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive" className="bg-red-500/10 border-red-500/30 text-red-300">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {!resumeText.trim() ? (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <AlertCircle className="h-5 w-5 text-yellow-400" />
              <p className="font-medium text-yellow-300">No Resume Text Available</p>
            </div>
            <p className="text-sm text-yellow-400 mb-3">
              Please upload a resume file first to use the AI refinement feature.
            </p>
            <p className="text-xs text-gray-400">
              Go to the "Upload" tab to get started.
            </p>
          </div>
        ) : (
          <Button 
            onClick={performRefinement} 
            disabled={isRefining}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white border-0"
          >
            {isRefining ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Refining Resume...
              </>
            ) : (
              <>
                <Wand2 className="mr-2 h-4 w-4" />
                Refine Resume with AI
              </>
            )}
          </Button>
        )}

        {refinedResume && (
          <Tabs defaultValue="refined" className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-gray-700/50 border border-gray-600">
              <TabsTrigger value="refined" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-300 text-gray-400">Refined Resume</TabsTrigger>
              <TabsTrigger value="improvements" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-300 text-gray-400">
                Improvements ({improvements.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="refined" className="space-y-4">
              <Card className="bg-gray-800/50 border-gray-600">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-white">
                    <span>Enhanced Resume</span>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={copyToClipboard}
                        className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-300 border-gray-600"
                      >
                        {copied ? (
                          <>
                            <CheckCircle className="h-4 w-4" />
                            Copied!
                          </>
                        ) : (
                          <>
                            <Copy className="h-4 w-4" />
                            Copy
                          </>
                        )}
                      </Button>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Textarea
                    value={refinedResume}
                    readOnly
                    className="min-h-[400px] font-mono text-sm bg-gray-900/50 border-gray-600 text-gray-300"
                    placeholder="Refined resume will appear here..."
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="improvements" className="space-y-4">
              {improvements.length > 0 ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">AI Improvements Made</h3>
                    <Badge variant="outline" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                      {improvements.length} improvements
                    </Badge>
                  </div>
                  
                  {improvements.map((improvement, index) => (
                    <Card key={index} className="bg-gray-800/50 border-gray-600">
                      <CardHeader>
                        <CardTitle className="flex items-center justify-between text-white">
                          <span className="capitalize">{improvement.type}</span>
                          <Badge variant={getImpactBadgeVariant(improvement.impact)} className={
                            improvement.impact.toLowerCase() === 'high' 
                              ? 'bg-green-500/20 text-green-300 border-green-500/30'
                              : improvement.impact.toLowerCase() === 'medium'
                              ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
                              : 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                          }>
                            {improvement.impact} Impact
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div>
                          <h4 className="font-medium text-red-400 mb-2">Original:</h4>
                          <p className="text-sm bg-red-500/10 border border-red-500/30 p-2 rounded border-l-4 border-red-400 text-gray-300">
                            {improvement.original}
                          </p>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-green-400 mb-2">Improved:</h4>
                          <p className="text-sm bg-green-500/10 border border-green-500/30 p-2 rounded border-l-4 border-green-400 text-gray-300">
                            {improvement.improved}
                          </p>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-blue-400 mb-2">Reason:</h4>
                          <p className="text-sm text-gray-400">{improvement.reason}</p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Card className="bg-gray-800/50 border-gray-600">
                  <CardContent className="text-center py-8">
                    <p className="text-gray-500">No improvements to display yet. Refine your resume to see AI enhancements.</p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
}