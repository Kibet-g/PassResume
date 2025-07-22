'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, Download, FileText, Palette, CheckCircle, AlertCircle } from 'lucide-react';

interface ResumeData {
  personal_info: {
    name: string;
    email: string;
    phone: string;
    address: string;
    linkedin?: string;
    website?: string;
  };
  summary: string;
  experience: Array<{
    title: string;
    company: string;
    duration: string;
    description: string;
  }>;
  education: Array<{
    degree: string;
    institution: string;
    year: string;
    gpa?: string;
  }>;
  skills: string[];
  certifications?: Array<{
    name: string;
    issuer: string;
    date: string;
  }>;
}

interface PDFExportProps {
  resumeText: string;
  refinedResume?: string;
  onExportComplete: (filename: string) => void;
}

const TEMPLATE_STYLES = [
  {
    id: 'professional',
    name: 'Professional',
    description: 'Clean, traditional layout perfect for corporate roles',
    preview: '📄'
  },
  {
    id: 'modern',
    name: 'Modern',
    description: 'Contemporary design with subtle colors and modern typography',
    preview: '🎨'
  },
  {
    id: 'executive',
    name: 'Executive',
    description: 'Sophisticated layout for senior-level positions',
    preview: '👔'
  }
];

export default function PDFExport({ resumeText, refinedResume, onExportComplete }: PDFExportProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState('professional');
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');

  const parseResumeData = (text: string): ResumeData => {
    // Simple parsing logic - in a real implementation, this would be more sophisticated
    const lines = text.split('\n').filter(line => line.trim());
    
    // Extract basic information (this is a simplified parser)
    const resumeData: ResumeData = {
      personal_info: {
        name: 'John Doe',
        email: 'john.doe@email.com',
        phone: '(555) 123-4567',
        address: 'City, State'
      },
      summary: '',
      experience: [],
      education: [],
      skills: []
    };

    // Extract name (usually first line)
    if (lines.length > 0) {
      resumeData.personal_info.name = lines[0];
    }

    // Extract email
    const emailMatch = text.match(/[\w\.-]+@[\w\.-]+\.\w+/);
    if (emailMatch) {
      resumeData.personal_info.email = emailMatch[0];
    }

    // Extract phone
    const phoneMatch = text.match(/\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/);
    if (phoneMatch) {
      resumeData.personal_info.phone = phoneMatch[0];
    }

    // Extract summary (look for summary/objective section)
    const summaryMatch = text.match(/(?:summary|objective)[:\s]*(.*?)(?=\n\n|\nexperience|\neducation|$)/i);
    if (summaryMatch) {
      resumeData.summary = summaryMatch[1].trim();
    }

    // Extract skills (look for skills section)
    const skillsMatch = text.match(/skills[:\s]*(.*?)(?=\n\n|\nexperience|\neducation|$)/i);
    if (skillsMatch) {
      resumeData.skills = skillsMatch[1].split(/[,\n]/).map(skill => skill.trim()).filter(Boolean);
    }

    return resumeData;
  };

  const exportToPDF = async () => {
    const textToExport = refinedResume || resumeText;
    
    if (!textToExport.trim()) {
      setError('Please provide resume content to export');
      return;
    }

    setIsExporting(true);
    setError('');
    setSuccess('');

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${apiUrl}/api/export-pdf-public`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          resume_text: textToExport,
          template_style: selectedTemplate 
        })
      });

      if (!response.ok) {
        throw new Error('Failed to export PDF');
      }

      const data = await response.json();
      
      // Convert base64 to blob and download
      const pdfBlob = new Blob([
        Uint8Array.from(atob(data.pdf_data), c => c.charCodeAt(0))
      ], { type: 'application/pdf' });
      
      const url = URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setSuccess(`PDF exported successfully as ${data.filename}`);
      onExportComplete(data.filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          PDF Export System
        </CardTitle>
        <CardDescription>
          Export your refined resume as a professional PDF with multiple template options
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

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Palette className="h-5 w-5" />
              Choose Template Style
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {TEMPLATE_STYLES.map((template) => (
                <Card 
                  key={template.id}
                  className={`cursor-pointer transition-all ${
                    selectedTemplate === template.id 
                      ? 'ring-2 ring-blue-500 bg-blue-50' 
                      : 'hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedTemplate(template.id)}
                >
                  <CardContent className="p-4 text-center">
                    <div className="text-3xl mb-2">{template.preview}</div>
                    <h4 className="font-semibold">{template.name}</h4>
                    <p className="text-sm text-gray-600 mt-1">{template.description}</p>
                    {selectedTemplate === template.id && (
                      <Badge className="mt-2">Selected</Badge>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          <div className="border-t pt-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="font-semibold">Export Options</h4>
                <p className="text-sm text-gray-600">
                  {refinedResume ? 'Exporting refined resume' : 'Exporting original resume'}
                </p>
              </div>
              <Badge variant={refinedResume ? 'default' : 'secondary'}>
                {refinedResume ? 'AI Enhanced' : 'Original'}
              </Badge>
            </div>

            <Button 
              onClick={exportToPDF} 
              disabled={isExporting || (!resumeText.trim() && !refinedResume)}
              className="w-full"
              size="lg"
            >
              {isExporting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating PDF...
                </>
              ) : (
                <>
                  <Download className="mr-2 h-4 w-4" />
                  Export as PDF ({TEMPLATE_STYLES.find(t => t.id === selectedTemplate)?.name})
                </>
              )}
            </Button>
          </div>

          <div className="text-xs text-gray-500 space-y-1">
            <p>• PDF will be optimized for ATS (Applicant Tracking Systems)</p>
            <p>• Original formatting and structure will be preserved</p>
            <p>• Professional fonts and layout will be applied</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}