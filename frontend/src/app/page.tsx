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
  Psychology,
  ViewInAr
} from '@mui/icons-material';
import { toast } from 'react-hot-toast';

// Import our Phase 3 components
import MedicalImageUpload from '../components/MedicalImageUpload';
import RealTimeAnalysisDashboard from '../components/RealTimeAnalysisDashboard';
import Medical3DViewer from '../components/Medical3DViewer';
import RealisticBrainViewer from '../components/RealisticBrainViewer';
import Advanced3DBrainViewer from '../components/Simple3DBrainViewer';
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
  const [visualizationData, setVisualizationData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStats, setConnectionStats] = useState({
    totalAnalyses: 0,
    successfulAnalyses: 0,
    averageProcessingTime: 0
  });

  /**
   * Handle analysis start
   */
  const handleAnalysisStart = (analysisId: string, files: any[], model: string) => {
    console.log('🚀 Analysis started:', { analysisId, files: files.length, model });
    setCurrentAnalysisId(analysisId);
    setActiveTab(1); // Switch to dashboard tab
    setConnectionStats(prev => ({
      ...prev,
      totalAnalyses: prev.totalAnalyses + 1
    }));
  };

  /**
   * Load enhanced 3D visualization data
   */
  const loadEnhanced3DData = async (analysisId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/analysis/${analysisId}/enhanced_3d`, {
        headers: {
          'Authorization': 'Bearer mock-token'
        }
      });
      
      if (response.ok) {
        const enhanced3DData = await response.json();
        console.log('🧠 Enhanced 3D data loaded:', enhanced3DData);
        return enhanced3DData;
      } else {
        console.error('❌ Failed to load enhanced 3D data:', response.statusText);
        return null;
      }
    } catch (error) {
      console.error('❌ Error loading enhanced 3D data:', error);
      return null;
    }
  };

  /**
   * Handle analysis completion
   */
  const handleAnalysisComplete = async (results: any) => {
    console.log('✅ Analysis completed:', results);
    setAnalysisResults(results);
    setConnectionStats(prev => ({
      ...prev,
      successfulAnalyses: prev.successfulAnalyses + 1
    }));
    
    // Load enhanced 3D data if analysis ID is available
    if (currentAnalysisId) {
      const enhanced3DData = await loadEnhanced3DData(currentAnalysisId);
      if (enhanced3DData) {
        setVisualizationData(enhanced3DData);
        toast.success('✅ Analysis complete! Enhanced 3D brain visualization loaded.', {
          duration: 8000
        });
      } else {
        toast.success('✅ Analysis complete! Medical report ready.');
      }
    } else {
      // Store visualization data for potential 3D viewing, but don't auto-redirect
      if (results?.[0]?.visualization_data) {
        setVisualizationData(results[0].visualization_data);
        toast.success('✅ Analysis complete! Medical report ready. 3D visualization available.', {
          duration: 8000
        });
      } else {
        toast.success('✅ Analysis complete! Medical report is ready.');
      }
    }
  };

  /**
   * Handle view 3D request from dashboard
   */
  const handleView3D = (visualizationData: any, analysis: any) => {
    console.log('🎯 View 3D requested for analysis:', analysis.analysis_id);
    setVisualizationData(visualizationData);
    setActiveTab(2); // Switch to 3D viewer tab
    toast.success('Switching to 3D viewer...', { duration: 3000 });
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
    });

    const initializeConnection = async () => {
      try {
        console.log('📡 Attempting WebSocket connection...');
        // Use a default user ID for demo purposes
        const userId = `user_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
        const connected = await websocket.connect(userId);
        console.log('✅ Connection result:', connected, 'for user:', userId);
        setIsConnected(connected);
        
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
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          🧠 Brain MRI Tumor Detector
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Phase 3: Real-time AI Analysis with 3D Visualization
        </Typography>
        
        {/* Status Cards */}
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <Psychology color={isConnected ? 'success' : 'error'} />
                  <Typography variant="h6">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  WebSocket Status
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
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
            <Card>
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
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <Chip 
                    label="v3.0" 
                    color="primary" 
                    size="small" 
                  />
                  <Typography variant="body2">
                    Phase 3
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  System Version
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Connection Status Alert */}
      {!isConnected && (
        <Alert 
          severity="warning" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => websocket.connect()}>
              Reconnect
            </Button>
          }
        >
          Not connected to the analysis server. Some features may be limited.
        </Alert>
      )}

      {/* Connection Status and Test */}
      <Box sx={{ mb: 3 }}>
        <Card variant="outlined">
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="h6">
                  WebSocket Status:
                </Typography>
                <Chip
                  icon={isConnected ? <Psychology /> : <Settings />}
                  label={isConnected ? 'Connected' : 'Disconnected'}
                  color={isConnected ? 'success' : 'warning'}
                  variant="outlined"
                />
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button 
                  variant="outlined" 
                  size="small" 
                  onClick={async () => {
                    try {
                      const userId = `test_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
                      console.log('🔄 Manual WebSocket connection test with user:', userId);
                      const connected = await websocket.connect(userId);
                      console.log('✅ Manual connection result:', connected);
                      setIsConnected(connected);
                    } catch (error) {
                      console.error('❌ Manual connection failed:', error);
                    }
                  }}
                >
                  Test Connection
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  color="error"
                  onClick={() => {
                    websocket.disconnect();
                    setIsConnected(false);
                  }}
                >
                  Disconnect
                </Button>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Debug Information */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2">
            <strong>Debug Info:</strong> Connection State = {isConnected ? 'Connected' : 'Disconnected'} | 
            WebSocket Ready State = {websocket.isConnected() ? 'OPEN' : 'NOT_OPEN'} | 
            User ID = {websocket.getUserId() || 'None'}
          </Typography>
          <Button 
            variant="contained" 
            size="small" 
            onClick={async () => {
              console.log('🔄 Manual reconnect triggered');
              websocket.disconnect();
              await new Promise(resolve => setTimeout(resolve, 1000));
              const result = await websocket.connect();
              console.log('🔄 Manual reconnect result:', result);
            }}
          >
            Force Reconnect
          </Button>
        </Box>
      </Alert>

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
            <Tab 
              label="3D Visualization" 
              icon={<ViewInAr />} 
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
            onView3D={handleView3D}
          />
        </TabPanel>

        {/* Tab Panel 2: 3D Visualization */}
        <TabPanel value={activeTab} index={2}>
          {visualizationData ? (
            <Box sx={{ height: '600px' }}>
              <Advanced3DBrainViewer
                data={visualizationData}
                analysisId={currentAnalysisId || undefined}
                onTumorClick={(tumorData) => {
                  console.log('Tumor clicked:', tumorData);
                  toast.success('Tumor details loaded');
                }}
              />
            </Box>
          ) : (
            <Alert severity="info" sx={{ m: 2 }}>
              <Typography variant="h6" gutterBottom>
                🧠 Advanced 3D Brain Visualization
              </Typography>
              <Typography variant="body2" gutterBottom>
                Upload and analyze medical images to generate advanced 3D brain visualizations.
                The enhanced viewer supports:
              </Typography>
              <Box component="ul" sx={{ mt: 1, pl: 2 }}>
                <li>External 3D brain models (GLB, GLTF, OBJ, FBX)</li>
                <li>Anatomically accurate medical visualization</li>
                <li>Tumor detection with interactive highlighting</li>
                <li>Clinical-grade lighting and materials</li>
                <li>Auto-detection of available brain models</li>
              </Box>
              <Box sx={{ mt: 2, height: '500px' }}>
                <Advanced3DBrainViewer
                  onTumorClick={(tumorData) => {
                    console.log('Demo tumor clicked:', tumorData);
                    toast('Demo tumor interaction');
                  }}
                />
              </Box>
            </Alert>
          )}
        </TabPanel>
      </Paper>

      {/* Footer */}
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Brain MRI Tumor Detector v3.0 - Phase 3: Advanced AI with Real-time WebSocket Integration & 3D Visualization
        </Typography>
        <Box sx={{ mt: 1 }}>
          <Chip label="🚀 Phase 3 Step 3 Complete" color="success" size="small" sx={{ mr: 1 }} />
          <Chip label="WebSocket Real-time" color="primary" size="small" sx={{ mr: 1 }} />
          <Chip label="3D Medical Viewer" color="secondary" size="small" sx={{ mr: 1 }} />
          <Chip label="6 AI Models" color="info" size="small" />
        </Box>
      </Box>
    </Container>
  );
}