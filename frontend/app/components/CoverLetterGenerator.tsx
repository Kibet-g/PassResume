'use client'

import React, { useState, useEffect } from 'react'
import { Button } from "./ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Textarea } from "./ui/textarea"
import { Badge } from "./ui/badge"
import { Alert, AlertDescription } from "./ui/alert"
import { Loader2, FileText, Download, Sparkles, Eye, Copy, CheckCircle, AlertCircle } from "lucide-react"

interface CoverLetterTemplate {
  name: string
  description: string
  style: string
}

interface CoverLetterGeneratorProps {
  resumeData?: any
  onGenerated?: (coverLetter: string) => void
}

export default function CoverLetterGenerator({ resumeData, onGenerated }: CoverLetterGeneratorProps) {
  const [jobDescription, setJobDescription] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState('professional')
  const [templates, setTemplates] = useState<CoverLetterTemplate[]>([])
  const [generatedCoverLetter, setGeneratedCoverLetter] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [alert, setAlert] = useState<{ type: 'success' | 'error', message: string } | null>(null)

  useEffect(() => {
    fetchTemplates()
  }, [])

  const fetchTemplates = async () => {
    try {
      const response = await fetch('/api/cover-letter/templates')
      if (response.ok) {
        const data = await response.json()
        setTemplates(data.templates || [])
      }
    } catch (error) {
      console.error('Error fetching templates:', error)
    }
  }

  const generateCoverLetter = async () => {
    if (!jobDescription.trim()) {
      setAlert({ type: 'error', message: 'Please enter a job description' })
      return
    }

    setAlert(null)
    setIsGenerating(true)
    try {
      const response = await fetch('/api/cover-letter/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_description: jobDescription,
          template_name: selectedTemplate,
          resume_data: resumeData
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setGeneratedCoverLetter(data.cover_letter)
        setShowPreview(true)
        onGenerated?.(data.cover_letter)
        setAlert({ type: 'success', message: 'Cover letter generated successfully!' })
      } else {
        const errorData = await response.json()
        setAlert({ type: 'error', message: errorData.error || 'Failed to generate cover letter' })
      }
    } catch (error) {
      console.error('Error generating cover letter:', error)
      setAlert({ type: 'error', message: 'Failed to generate cover letter' })
    } finally {
      setIsGenerating(false)
    }
  }

  const exportToPDF = async () => {
    if (!generatedCoverLetter) {
      setAlert({ type: 'error', message: 'No cover letter to export' })
      return
    }

    setAlert(null)
    setIsExporting(true)
    try {
      const response = await fetch('/api/cover-letter/export-pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cover_letter_text: generatedCoverLetter,
          template_name: selectedTemplate
        }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.style.display = 'none'
        a.href = url
        a.download = `cover_letter_${selectedTemplate}_${new Date().toISOString().split('T')[0]}.pdf`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
        setAlert({ type: 'success', message: 'Cover letter exported successfully!' })
      } else {
        const errorData = await response.json()
        setAlert({ type: 'error', message: errorData.error || 'Failed to export cover letter' })
      }
    } catch (error) {
      console.error('Error exporting cover letter:', error)
      setAlert({ type: 'error', message: 'Failed to export cover letter' })
    } finally {
      setIsExporting(false)
    }
  }

  const copyToClipboard = async () => {
    if (!generatedCoverLetter) return
    
    try {
      await navigator.clipboard.writeText(generatedCoverLetter)
      setAlert({ type: 'success', message: 'Cover letter copied to clipboard!' })
    } catch (error) {
      setAlert({ type: 'error', message: 'Failed to copy to clipboard' })
    }
  }

  return (
    <div className="space-y-6">
      {alert && (
        <Alert variant={alert.type === 'error' ? 'destructive' : 'default'}>
          {alert.type === 'error' ? (
            <AlertCircle className="h-4 w-4" />
          ) : (
            <CheckCircle className="h-4 w-4" />
          )}
          <AlertDescription>{alert.message}</AlertDescription>
        </Alert>
      )}
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-blue-600" />
            AI Cover Letter Generator
          </CardTitle>
          <CardDescription>
            Generate a tailored cover letter based on your resume and job description
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">
              Job Description
            </label>
            <Textarea
              placeholder="Paste the job description here..."
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              rows={6}
              className="resize-none"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">
              Template Style
            </label>
            <select 
              value={selectedTemplate} 
              onChange={(e) => setSelectedTemplate(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {templates.map((template) => (
                <option key={template.name} value={template.name}>
                  {template.name} - {template.description}
                </option>
              ))}
            </select>
          </div>

          <Button 
            onClick={generateCoverLetter} 
            disabled={isGenerating || !jobDescription.trim()}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Cover Letter
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {generatedCoverLetter && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-green-600" />
                Generated Cover Letter
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowPreview(!showPreview)}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  {showPreview ? 'Hide' : 'Preview'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyToClipboard}
                >
                  <Copy className="h-4 w-4 mr-1" />
                  Copy
                </Button>
                <Button
                  size="sm"
                  onClick={exportToPDF}
                  disabled={isExporting}
                >
                  {isExporting ? (
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4 mr-1" />
                  )}
                  Export PDF
                </Button>
              </div>
            </CardTitle>
            <CardDescription>
              <Badge variant="secondary" className="capitalize">
                {selectedTemplate} Template
              </Badge>
            </CardDescription>
          </CardHeader>
          {showPreview && (
            <CardContent>
              <div className="bg-gray-50 p-4 rounded-lg border">
                <pre className="whitespace-pre-wrap text-sm font-mono">
                  {generatedCoverLetter}
                </pre>
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  )
}