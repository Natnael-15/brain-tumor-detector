import type { Metadata, Viewport } from 'next';
import ClientLayout from './ClientLayout';
import './globals.css';

export const metadata: Metadata = {
  title: 'Brain MRI Tumor Detector - Phase 3',
  description: 'Advanced AI-powered brain tumor detection and analysis with real-time WebSocket updates and 3D visualization',
  keywords: ['brain tumor', 'MRI', 'AI', 'medical imaging', 'tumor detection', 'deep learning'],
  authors: [{ name: 'Brain Tumor Detector Team' }],
  icons: {
    icon: '/favicon.ico',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="theme-color" content="#1976d2" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
      </head>
      <body suppressHydrationWarning>
        <ClientLayout>
          {children}
        </ClientLayout>
      </body>
    </html>
  );
}