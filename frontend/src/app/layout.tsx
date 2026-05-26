import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'E-Nose Prostate Cancer Predictor',
  description: 'Explainable AI Clinical Support Dashboard for Prostate Cancer screening using Volatile Organic Compound sensor signal arrays.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen text-slate-100 bg-[#05070c]">
        {children}
      </body>
    </html>
  )
}
