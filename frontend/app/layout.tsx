import './globals.css'
import { Inter } from 'next/font/google'
import { AuthProvider } from './contexts/AuthContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Open Resume Auditor - Free ATS Resume Checker',
  description: 'Free AI-powered resume auditing tool that ensures ATS compliance and increases your chances of landing interviews.',
  keywords: 'resume, ATS, job search, career, AI, free, auditor',
  authors: [{ name: 'Open Resume Auditor Team' }],
  openGraph: {
    title: 'Open Resume Auditor - Free ATS Resume Checker',
    description: 'Free AI-powered resume auditing tool that ensures ATS compliance and increases your chances of landing interviews.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
            {children}
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}