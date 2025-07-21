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
      const token = localStorage.getItem('token');
      const response = await fetch('/api/refine-resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
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
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Wand2 className="h-5 w-5" />
          AI Resume Refiner
        </CardTitle>
        <CardDescription>
          Intelligent resume enhancement with AI-powered improvements and optimization
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Button 
          onClick={performRefinement} 
          disabled={isRefining || !resumeText.trim()}
          className="w-full"
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

        {refinedResume && (
          <Tabs defaultValue="refined" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="refined">Refined Resume</TabsTrigger>
              <TabsTrigger value="improvements">
                Improvements ({improvements.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="refined" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Enhanced Resume</span>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={copyToClipboard}
                        className="flex items-center gap-2"
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
                    className="min-h-[400px] font-mono text-sm"
                    placeholder="Refined resume will appear here..."
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="improvements" className="space-y-4">
              {improvements.length > 0 ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">AI Improvements Made</h3>
                    <Badge variant="outline">
                      {improvements.length} improvements
                    </Badge>
                  </div>
                  
                  {improvements.map((improvement, index) => (
                    <Card key={index}>
                      <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                          <span className="capitalize">{improvement.type}</span>
                          <Badge variant={getImpactBadgeVariant(improvement.impact)}>
                            {improvement.impact} Impact
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div>
                          <h4 className="font-medium text-red-600 mb-2">Original:</h4>
                          <p className="text-sm bg-red-50 p-2 rounded border-l-4 border-red-200">
                            {improvement.original}
                          </p>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-green-600 mb-2">Improved:</h4>
                          <p className="text-sm bg-green-50 p-2 rounded border-l-4 border-green-200">
                            {improvement.improved}
                          </p>
                        </div>
                        
                        <div>
                          <h4 className="font-medium text-blue-600 mb-2">Reason:</h4>
                          <p className="text-sm text-gray-700">{improvement.reason}</p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <Card>
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