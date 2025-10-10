import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import ClientLayout from './ClientLayout';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Brain MRI Tumor Detector - Phase 3',
  description: 'Advanced automated brain tumor detection and analysis with real-time WebSocket updates and 3D visualization',
  keywords: ['brain tumor', 'MRI', 'medical imaging', 'tumor detection', 'deep learning', 'automated diagnosis'],
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
      <body className={inter.className} suppressHydrationWarning>
        <ClientLayout>
          {children}
        </ClientLayout>
      </body>
    </html>
  );
}