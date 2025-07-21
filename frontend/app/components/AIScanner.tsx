'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Loader2, Brain, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react';

interface ScanResult {
  section: string;
  quality_score: number;
  issues: string[];
  suggestions: string[];
  keywords_found: string[];
  missing_keywords: string[];
}

interface AIScannerProps {
  resumeText: string;
  onScanComplete: (results: any) => void;
}

export default function AIScanner({ resumeText, onScanComplete }: AIScannerProps) {
  const [isScanning, setIsScanning] = useState(false);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [qualityScore, setQualityScore] = useState<number>(0);
  const [error, setError] = useState<string>('');

  const performIntelligentScan = async () => {
    if (!resumeText.trim()) {
      setError('Please provide resume text to scan');
      return;
    }

    setIsScanning(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/intelligent-scan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ resume_text: resumeText })
      });

      if (!response.ok) {
        throw new Error('Failed to perform intelligent scan');
      }

      const data = await response.json();
      setScanResults(data.scan_results);
      setQualityScore(data.quality_score);
      onScanComplete(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed');
    } finally {
      setIsScanning(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          AI Resume Scanner
        </CardTitle>
        <CardDescription>
          Advanced AI analysis of your resume sections and quality assessment
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
          onClick={performIntelligentScan} 
          disabled={isScanning || !resumeText.trim()}
          className="w-full"
        >
          {isScanning ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Scanning Resume...
            </>
          ) : (
            <>
              <Brain className="mr-2 h-4 w-4" />
              Start AI Scan
            </>
          )}
        </Button>

        {qualityScore > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Overall Quality Score</span>
                <Badge variant={getScoreBadgeVariant(qualityScore)}>
                  {qualityScore}/100
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    qualityScore >= 80 ? 'bg-green-500' : 
                    qualityScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${qualityScore}%` }}
                ></div>
              </div>
              <p className={`text-sm mt-2 ${getScoreColor(qualityScore)}`}>
                {qualityScore >= 80 ? 'Excellent resume quality!' :
                 qualityScore >= 60 ? 'Good resume, room for improvement' :
                 'Resume needs significant improvements'}
              </p>
            </CardContent>
          </Card>
        )}

        {scanResults.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Section Analysis
            </h3>
            {scanResults.map((result, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="capitalize">{result.section}</span>
                    <Badge variant={getScoreBadgeVariant(result.quality_score)}>
                      {result.quality_score}/100
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {result.issues.length > 0 && (
                    <div>
                      <h4 className="font-medium text-red-600 mb-2">Issues Found:</h4>
                      <ul className="list-disc list-inside space-y-1">
                        {result.issues.map((issue, i) => (
                          <li key={i} className="text-sm text-red-600">{issue}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.suggestions.length > 0 && (
                    <div>
                      <h4 className="font-medium text-blue-600 mb-2">Suggestions:</h4>
                      <ul className="list-disc list-inside space-y-1">
                        {result.suggestions.map((suggestion, i) => (
                          <li key={i} className="text-sm text-blue-600">{suggestion}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.keywords_found.length > 0 && (
                    <div>
                      <h4 className="font-medium text-green-600 mb-2">Keywords Found:</h4>
                      <div className="flex flex-wrap gap-1">
                        {result.keywords_found.map((keyword, i) => (
                          <Badge key={i} variant="outline" className="text-green-600">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.missing_keywords.length > 0 && (
                    <div>
                      <h4 className="font-medium text-orange-600 mb-2">Missing Keywords:</h4>
                      <div className="flex flex-wrap gap-1">
                        {result.missing_keywords.map((keyword, i) => (
                          <Badge key={i} variant="outline" className="text-orange-600">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}