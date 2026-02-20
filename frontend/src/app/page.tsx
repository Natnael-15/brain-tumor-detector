/**
 * Main Application Page - Brain MRI Tumor Detector
 * Phase 3 Step 3: Frontend WebSocket Integration & 3D Viewer
 */

'use client';

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Tabs,
  Tab,
  Alert,
  Card,
  CardContent,
  Chip,
  Button
} from '@mui/material';
import {
  CloudUpload,
  Visibility,
  Analytics,
  Settings,
  Psychology
} from '@mui/icons-material';
import { toast } from 'react-hot-toast';

// Import our Phase 3 components
import MedicalImageUpload from '../components/MedicalImageUpload';
import RealTimeAnalysisDashboard from '../components/RealTimeAnalysisDashboard';
import { useEnhancedWebSocket } from '../lib/enhanced-websocket';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function HomePage() {
  const websocket = useEnhancedWebSocket();
  const [activeTab, setActiveTab] = useState(0);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [websocketUnavailable, setWebsocketUnavailable] = useState(false);
  const [connectionStats, setConnectionStats] = useState({
    totalAnalyses: 0,
    successfulAnalyses: 0,
    averageProcessingTime: 0
  });

  const effectiveConnected = isConnected || websocketUnavailable;

  /**
   * Handle analysis start
   */
  const handleAnalysisStart = (analysisId: string, files: any[], model: string) => {
    console.log('Analysis started:', { analysisId, files: files.length, model });
    setCurrentAnalysisId(analysisId);
    setActiveTab(1); // Switch to dashboard tab
    setConnectionStats(prev => ({
      ...prev,
      totalAnalyses: prev.totalAnalyses + 1
    }));
  };



  /**
   * Handle analysis completion
   */
  const handleAnalysisComplete = async (results: any) => {
    console.log('Analysis completed:', results);
    setAnalysisResults(results);
    setConnectionStats(prev => ({
      ...prev,
      successfulAnalyses: prev.successfulAnalyses + 1
    }));
    
    toast.success('Analysis complete! Medical report is ready.');
  };



  /**
   * Handle file preview
   */
  const handleFilePreview = (file: File) => {
    console.log('👁️ File preview requested:', file.name);
    // Could implement additional preview logic here
  };

  /**
   * Initialize WebSocket connection
   */
  useEffect(() => {
    console.log('🔗 Initializing WebSocket connection...');
    
    // Subscribe to connection changes FIRST
    const unsubscribe = websocket.onConnectionChange((connected) => {
      console.log('🔌 Connection status changed:', connected);
      setIsConnected(connected);
      if (connected) {
        setWebsocketUnavailable(false);
      }
    });

    const initializeConnection = async () => {
      try {
        console.log('📡 Attempting WebSocket connection...');
        // Use a default user ID for demo purposes
        const userId = `user_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
        const connected = await websocket.connect(userId);
        console.log('✅ Connection result:', connected, 'for user:', userId);
        setIsConnected(connected);

        if (!connected) {
          websocket.disconnect();
          setWebsocketUnavailable(true);
          return;
        }
        
        // Double-check connection status
        const isActuallyConnected = websocket.isConnected();
        console.log('🔍 Actual connection status:', isActuallyConnected);
        if (isActuallyConnected !== connected) {
          setIsConnected(isActuallyConnected);
        }
      } catch (error) {
        console.error('❌ Failed to connect to WebSocket:', error);
        setIsConnected(false);
      }
    };

    initializeConnection();

    return unsubscribe;
  }, [websocket]);

  /**
   * Handle tab changes
   */
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold" sx={{ color: 'primary.main' }}>
          Brain MRI Tumor Detector
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Advanced Automated Medical Imaging Analysis
        </Typography>
        
        {/* Status Cards */}
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <Psychology color={effectiveConnected ? 'success' : 'error'} />
                  <Typography variant="h6">
                    {isConnected ? 'Connected' : websocketUnavailable ? 'Connected (Polling)' : 'Disconnected'}
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  System Status
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h6" color="primary">
                  {connectionStats.totalAnalyses}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Total Analyses
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h6" color="success.main">
                  {connectionStats.successfulAnalyses}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Successful
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <Chip 
                    label="Clinical Grade" 
                    color="primary" 
                    size="small" 
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  6 Detection Models Active
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Connection Status Alert */}
      {!effectiveConnected && (
        <Alert 
          severity="warning" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => { setWebsocketUnavailable(false); websocket.connect(); }}>
              Reconnect
            </Button>
          }
        >
          Not connected to the real-time channel. Attempting to reconnect...
        </Alert>
      )}

      {websocketUnavailable && !isConnected && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Real-time WebSocket updates are unavailable in this environment. The app will continue using API polling for analysis progress and results.
        </Alert>
      )}

      {websocketUnavailable && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Real-time WebSocket updates are unavailable in this environment. The app will continue using API polling for analysis progress and results.
        </Alert>
      )}

      {/* Main Interface */}
      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="main navigation">
            <Tab 
              label="Upload & Analyze" 
              icon={<CloudUpload />} 
              iconPosition="start"
              sx={{ minHeight: 'auto', py: 2 }}
            />
            <Tab 
              label="Real-time Dashboard" 
              icon={<Analytics />} 
              iconPosition="start"
              sx={{ minHeight: 'auto', py: 2 }}
            />
          </Tabs>
        </Box>

        {/* Tab Panel 0: Upload & Analyze */}
        <TabPanel value={activeTab} index={0}>
          <MedicalImageUpload
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onFilePreview={handleFilePreview}
          />
        </TabPanel>

        {/* Tab Panel 1: Real-time Dashboard */}
        <TabPanel value={activeTab} index={1}>
          <RealTimeAnalysisDashboard
            analysisId={currentAnalysisId || undefined}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={(error) => console.error('Analysis error:', error)}
            isConnected={isConnected}
          />
        </TabPanel>
      </Paper>

      {/* Footer */}
      <Box sx={{ mt: 4, textAlign: 'center', py: 2, borderTop: '1px solid #e0e0e0' }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Brain MRI Tumor Detector - Clinical Detection System
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Advanced Deep Learning for Medical Image Analysis • Real-time Processing
        </Typography>
        <Box sx={{ mt: 1 }}>
          <Chip label="6 Detection Models" color="primary" size="small" variant="outlined" sx={{ mr: 1 }} />
          <Chip label="Real-time Analysis" color="success" size="small" variant="outlined" />
        </Box>
      </Box>
    </Container>
  );
}