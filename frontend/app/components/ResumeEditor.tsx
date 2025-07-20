'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { cn } from '../../lib/utils';
import { 
  Edit3, 
  Save, 
  RefreshCw, 
  TrendingUp, 
  Target, 
  FileText, 
  Lightbulb,
  Star,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';

interface EditSuggestion {
  type: string;
  suggestion: string;
  keyword?: string;
  importance?: number;
  example?: string;
}

interface ResumeEditorProps {
  resumeId: number;
  originalText: string;
  onSave: (editedText: string) => void;
}

export default function ResumeEditor({ resumeId, originalText, onSave }: ResumeEditorProps) {
  const [editedText, setEditedText] = useState(originalText);
  const [suggestions, setSuggestions] = useState<EditSuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('edit');
  const [improvementScore, setImprovementScore] = useState(0);
  const [appliedSuggestions, setAppliedSuggestions] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchEditSuggestions('general');
  }, [resumeId]);

  const fetchEditSuggestions = async (editType: string) => {
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/edit-resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resume_id: resumeId,
          edit_type: editType
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSuggestions(data.suggestions || []);
      }
    } catch (error) {
      console.error('Error fetching edit suggestions:', error);
    } finally {
      setLoading(false);
    }
  };

  const applySuggestion = (suggestion: EditSuggestion, index: number) => {
    let newText = editedText;
    
    if (suggestion.type === 'keyword_addition' && suggestion.keyword) {
      // Smart keyword insertion
      const skillsSection = newText.match(/(SKILLS|Skills|TECHNICAL SKILLS)[\s\S]*?(?=\n[A-Z]|\n\n|$)/i);
      if (skillsSection) {
        newText = newText.replace(skillsSection[0], skillsSection[0] + `, ${suggestion.keyword}`);
      } else {
        newText += `\n\nSKILLS\n• ${suggestion.keyword}`;
      }
    } else if (suggestion.type === 'formatting' && suggestion.example) {
      // Add example formatting
      newText += `\n\n${suggestion.example}`;
    } else if (suggestion.type === 'structure' && suggestion.example) {
      // Add section headers
      newText = `${suggestion.example}\n\n${newText}`;
    }
    
    setEditedText(newText);
    setAppliedSuggestions(prev => new Set([...prev, index]));
    calculateImprovementScore(newText);
  };

  const calculateImprovementScore = (text: string) => {
    const originalWords = originalText.split(' ').length;
    const newWords = text.split(' ').length;
    const wordImprovement = Math.min((newWords - originalWords) / originalWords * 100, 20);
    
    const hasMoreBullets = (text.match(/[•\-\*]/g) || []).length > (originalText.match(/[•\-\*]/g) || []).length;
    const bulletImprovement = hasMoreBullets ? 15 : 0;
    
    const appliedCount = appliedSuggestions.size;
    const suggestionImprovement = (appliedCount / suggestions.length) * 30;
    
    const totalImprovement = Math.min(wordImprovement + bulletImprovement + suggestionImprovement, 100);
    setImprovementScore(Math.round(totalImprovement));
  };

  const handleSave = () => {
    onSave(editedText);
  };

  const resetText = () => {
    setEditedText(originalText);
    setAppliedSuggestions(new Set());
    setImprovementScore(0);
  };

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'keyword_addition': return <Target className="w-4 h-4" />;
      case 'formatting': return <FileText className="w-4 h-4" />;
      case 'structure': return <Edit3 className="w-4 h-4" />;
      case 'content_expansion': return <TrendingUp className="w-4 h-4" />;
      default: return <Lightbulb className="w-4 h-4" />;
    }
  };

  const getSuggestionColor = (importance?: number) => {
    if (!importance) return 'bg-blue-100 text-blue-800';
    if (importance > 0.7) return 'bg-red-100 text-red-800';
    if (importance > 0.4) return 'bg-yellow-100 text-yellow-800';
    return 'bg-green-100 text-green-800';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Edit3 className="w-5 h-5" />
            AI-Powered Resume Editor
          </CardTitle>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Improvement Score: {improvementScore}%
            </Badge>
            <Badge variant="outline">
              Applied: {appliedSuggestions.size}/{suggestions.length} suggestions
            </Badge>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="edit">Edit Resume</TabsTrigger>
          <TabsTrigger value="keywords">Keywords</TabsTrigger>
          <TabsTrigger value="formatting">Formatting</TabsTrigger>
          <TabsTrigger value="content">Content</TabsTrigger>
        </TabsList>

        <TabsContent value="edit" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Resume Editor</CardTitle>
                <div className="flex gap-2">
                  <Button onClick={handleSave} className="flex items-center gap-1">
                    <Save className="w-4 h-4" />
                    Save Changes
                  </Button>
                  <Button variant="outline" onClick={resetText}>
                    <RefreshCw className="w-4 h-4" />
                    Reset
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={editedText}
                  onChange={(e) => {
                    setEditedText(e.target.value);
                    calculateImprovementScore(e.target.value);
                  }}
                  className="min-h-[500px] font-mono text-sm"
                  placeholder="Edit your resume here..."
                />
                <div className="mt-2 text-sm text-gray-500">
                  Words: {editedText.split(' ').length} | Characters: {editedText.length}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">AI Suggestions</CardTitle>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => fetchEditSuggestions('keywords')}
                    disabled={loading}
                  >
                    Keywords
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => fetchEditSuggestions('formatting')}
                    disabled={loading}
                  >
                    Formatting
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => fetchEditSuggestions('content')}
                    disabled={loading}
                  >
                    Content
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center py-8">
                    <RefreshCw className="w-6 h-6 animate-spin" />
                    <span className="ml-2">Generating suggestions...</span>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {suggestions.length === 0 ? (
                      <Alert>
                        <CheckCircle className="w-4 h-4" />
                        <AlertDescription>
                          Great! No immediate suggestions. Your resume looks good.
                        </AlertDescription>
                      </Alert>
                    ) : (
                      suggestions.map((suggestion, index) => (
                        <Card key={index} className="border-l-4 border-l-blue-500">
                          <CardContent className="pt-4">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  {getSuggestionIcon(suggestion.type)}
                                  <Badge 
                                    variant="secondary" 
                                    className={getSuggestionColor(suggestion.importance)}
                                  >
                                    {suggestion.type.replace('_', ' ').toUpperCase()}
                                  </Badge>
                                  {suggestion.importance && (
                                    <div className="flex items-center">
                                      {[...Array(5)].map((_, i) => (
                                        <Star
                                          key={i}
                                          className={`w-3 h-3 ${
                                            i < suggestion.importance! * 5
                                              ? 'text-yellow-400 fill-current'
                                              : 'text-gray-300'
                                          }`}
                                        />
                                      ))}
                                    </div>
                                  )}
                                </div>
                                <p className="text-sm text-gray-700 mb-2">
                                  {suggestion.suggestion}
                                </p>
                                {suggestion.example && (
                                  <div className="bg-gray-50 p-2 rounded text-xs font-mono">
                                    {suggestion.example}
                                  </div>
                                )}
                              </div>
                              <Button
                                size="sm"
                                variant={appliedSuggestions.has(index) ? "secondary" : "default"}
                                onClick={() => applySuggestion(suggestion, index)}
                                disabled={appliedSuggestions.has(index)}
                                className="ml-2"
                              >
                                {appliedSuggestions.has(index) ? (
                                  <>
                                    <CheckCircle className="w-3 h-3 mr-1" />
                                    Applied
                                  </>
                                ) : (
                                  'Apply'
                                )}
                              </Button>
                            </div>
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="keywords">
          <Card>
            <CardHeader>
              <CardTitle>Keyword Optimization</CardTitle>
            </CardHeader>
            <CardContent>
              <Button onClick={() => fetchEditSuggestions('keywords')}>
                Get Keyword Suggestions
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="formatting">
          <Card>
            <CardHeader>
              <CardTitle>Formatting Improvements</CardTitle>
            </CardHeader>
            <CardContent>
              <Button onClick={() => fetchEditSuggestions('formatting')}>
                Get Formatting Suggestions
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="content">
          <Card>
            <CardHeader>
              <CardTitle>Content Enhancement</CardTitle>
            </CardHeader>
            <CardContent>
              <Button onClick={() => fetchEditSuggestions('content')}>
                Get Content Suggestions
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {improvementScore > 0 && (
        <Alert className="border-green-200 bg-green-50">
          <TrendingUp className="w-4 h-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Great progress! Your resume has improved by {improvementScore}%. 
            {improvementScore > 50 && " You're on track for better ATS compatibility!"}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}