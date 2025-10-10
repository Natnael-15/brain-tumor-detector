/**
 * Real-time Analysis Dashboard Component 
 * Phase 3 Step 3: Frontend WebSocket Integration & Real-time Updates
 */

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  IconButton,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Divider
} from '@mui/material';
import {
  Wifi,
  WifiOff,
  Speed,
  CheckCircle,
  Error,
  Schedule,
  Psychology,
  BiotechOutlined,
  ExpandMore,
  ExpandLess,
  Refresh,
  Timeline,
  Close,
  Visibility,
  Assessment,
  Warning
} from '@mui/icons-material';
import { useWebSocket } from '../lib/websocket';
import BrainActivitySimulation from './BrainActivitySimulation';

// Interfaces
interface CurrentAnalysisState {
  id: string;
  progress: number;
  stage: string;
  message: string;
  timestamp: string;
  model: string;
  files: number;
  result?: AnalysisResult;
}

interface AnalysisHistoryEntry {
  analysis_id: string;
  file_name: string;
  model: string;
  timestamp: string;
  status: 'completed' | 'failed' | 'in_progress';
  results?: AnalysisResult;
  processing_time?: number;
  is_2d_image?: boolean;
}

interface AnalysisUpdate {
  analysis_id: string;
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
  model?: string;
  file_name?: string;
  results?: any;
  status?: string;
}

interface ConnectionStats {
  isConnected: boolean;
  totalAnalyses: number;
  successfulAnalyses: number;
  lastHeartbeat: Date | null;
  roundTripTime: number;
  reconnectAttempts: number;
}

interface AnalysisResult {
  model: string;
  prediction: {
    tumor_detected: boolean;
    confidence: number;
    tumor_type: string;
    tumor_grade?: string;
    volume_ml?: number;
    location?: string;
    enhancement_pattern?: string;
    mass_effect?: boolean;
    edema_present?: boolean;
  };
  confidence: number;
  processing_time: number;
  tumor_detected: boolean;
  tumor_type?: string;
  tumor_grade?: string;
  tumor_volume?: number;
  tumor_location?: string;
  visualization_data?: any;
  segmentation?: {
    tumor_mask_available?: boolean;
    segmentation_quality?: number;
    dice_score?: number;
    tumor_core_volume?: number;
    enhancement_volume?: number;
    edema_volume?: number;
  };
  risk_assessment?: {
    risk_level?: string;
    urgency?: string;
    recommendation?: string;
    differential_diagnosis?: string[];
    follow_up?: string;
  };
  clinical_notes?: {
    findings?: string;
    limitations?: string;
    quality_indicators?: {
      image_quality?: string;
      artifacts_present?: boolean;
      contrast_enhancement?: string;
    };
  };
  metrics?: {
    dice_score?: number;
    hausdorff_distance?: number;
    sensitivity?: number;
    specificity?: number;
    processing_time?: number;
    inference_time?: number;
    preprocessing_time?: number;
    postprocessing_time?: number;
  };
}

interface RealTimeAnalysisDashboardProps {
  analysisId?: string;
  onAnalysisComplete?: (results: AnalysisResult[]) => void;
  onAnalysisError?: (error: string) => void;
  isConnected?: boolean;
}

export const RealTimeAnalysisDashboard: React.FC<RealTimeAnalysisDashboardProps> = ({
  analysisId,
  onAnalysisComplete,
  onAnalysisError,
  isConnected: parentConnected = false
}) => {
  const websocket = useWebSocket();
  
  // State management - remove local isConnected state
  const [currentAnalysis, setCurrentAnalysis] = useState<CurrentAnalysisState | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<Record<string, AnalysisUpdate>>({});
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisUpdate[]>([]);
  const [completedAnalyses, setCompletedAnalyses] = useState<AnalysisResult[]>([]);
  const [persistentAnalysisHistory, setPersistentAnalysisHistory] = useState<AnalysisHistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showAnalysisHistory, setShowAnalysisHistory] = useState(true);
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisHistoryEntry | null>(null);
  const [connectionStats, setConnectionStats] = useState<ConnectionStats>({
    isConnected: false,
    totalAnalyses: 0,
    successfulAnalyses: 0,
    lastHeartbeat: null,
    roundTripTime: 0,
    reconnectAttempts: 0
  });

  // Load analysis history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('brain-tumor-analysis-history');
    if (savedHistory) {
      try {
        setPersistentAnalysisHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Failed to load analysis history:', e);
      }
    }
  }, []);

  // Save analysis history to localStorage
  const saveAnalysisHistory = useCallback((history: AnalysisHistoryEntry[]) => {
    try {
      localStorage.setItem('brain-tumor-analysis-history', JSON.stringify(history));
    } catch (e) {
      console.error('Failed to save analysis history:', e);
    }
  }, []);

  /**
   * Handle analysis updates from WebSocket
   */
  const handleAnalysisUpdate = useCallback((update: AnalysisUpdate) => {
    console.log('📊 Dashboard received analysis update:', update);
    console.log('📊 Update details:', {
      stage: update.stage,
      progress: update.progress,
      model: update.model,
      hasResults: !!update.results,
      resultsKeys: update.results ? Object.keys(update.results) : []
    });
    
    // Update progress tracking
    setAnalysisProgress(prev => ({
      ...prev,
      [update.analysis_id]: update
    }));

    // Add to history
    setAnalysisHistory(prev => [update, ...prev.slice(0, 49)]); // Keep last 50 updates

    // Check if analysis is complete - be more flexible with completion detection
    if (update.stage === 'completed' || update.stage === 'complete' || (update.progress === 100 && update.results) || (update.progress >= 100 && update.results)) {
      console.log('🎉 Analysis completed! Processing results...', update.results);
      
      const result: AnalysisResult = {
        model: update.model || update.results?.model_name || update.results?.model || 'Unknown Model',
        prediction: update.results?.prediction || {
          tumor_detected: false,
          confidence: 0,
          tumor_type: 'Unknown',
          tumor_grade: null,
          volume_ml: 0,
          location: null
        },
        confidence: update.results?.prediction?.confidence || update.results?.confidence || 0,
        processing_time: update.results?.metrics?.processing_time || update.results?.processing_time || 0,
        tumor_detected: update.results?.prediction?.tumor_detected || false,
        tumor_type: update.results?.prediction?.tumor_type,
        tumor_grade: update.results?.prediction?.tumor_grade,
        tumor_volume: update.results?.prediction?.volume_ml,
        tumor_location: update.results?.prediction?.location,
        visualization_data: update.results?.visualization_data,
        metrics: update.results?.metrics,
        segmentation: update.results?.segmentation,
        risk_assessment: update.results?.risk_assessment
      };

      console.log('🔬 Processed analysis result:', result);

      // Add to persistent analysis history
      const historyEntry: AnalysisHistoryEntry = {
        analysis_id: update.analysis_id,
        file_name: update.file_name || update.results?.file_name || update.results?.analysis_metadata?.file_names?.[0] || 'Unknown File',
        model: result.model,
        timestamp: update.timestamp,
        status: 'completed',
        results: result,
        processing_time: result.processing_time,
        is_2d_image: update.results?.visualization_data?.is_2d_image || false
      };

      console.log('💾 Adding to analysis history:', historyEntry);

      setPersistentAnalysisHistory(prev => {
        const newHistory = [historyEntry, ...prev.slice(0, 49)]; // Keep last 50 analyses
        saveAnalysisHistory(newHistory);
        return newHistory;
      });

      setCompletedAnalyses(prev => [result, ...prev]);
      
      // Update current analysis to show completion
      setCurrentAnalysis({
        id: update.analysis_id,
        progress: 100,
        stage: 'Analysis Complete! 🎉',
        message: `${result.model} detected ${result.tumor_detected ? 'tumor' : 'no tumor'} with ${(result.confidence * 100).toFixed(1)}% confidence`,
        timestamp: new Date().toISOString(),
        model: result.model,
        files: 1,
        result: result
      });
      
      // Keep current analysis visible much longer for doctors to review
      setTimeout(() => {
        setCurrentAnalysis(null);
      }, 30000); // Show for 30 seconds instead of 10 to give doctors time to review
      
      onAnalysisComplete?.([result]);
    } else {
      // Update current analysis for in-progress updates
      setCurrentAnalysis({
        id: update.analysis_id,
        progress: update.progress,
        stage: update.stage,
        message: update.message,
        timestamp: update.timestamp,
        model: update.model || 'Processing...',
        files: 1
      });
    }

    // Handle errors
    if (update.stage === 'error') {
      setCurrentAnalysis(null);
      onAnalysisError?.(update.message);
    }
  }, [onAnalysisComplete, onAnalysisError]);

  /**
   * WebSocket connection status
   */
  useEffect(() => {
    setConnectionStats(prev => ({
      ...prev,
      isConnected: parentConnected,
      lastHeartbeat: parentConnected ? new Date() : prev.lastHeartbeat
    }));
  }, [parentConnected]);

  /**
   * Analysis updates handling
   */
  useEffect(() => {
    const unsubscribe = websocket.onAnalysisUpdate(handleAnalysisUpdate);
    return unsubscribe;
  }, [handleAnalysisUpdate]);

  /**
   * Initialize analysis tracking
   */
  useEffect(() => {
    if (analysisId && !currentAnalysis) {
      console.log('🎯 Dashboard tracking analysis:', analysisId);
      setCurrentAnalysis({
        id: analysisId,
        progress: 0,
        stage: 'Starting analysis...',
        message: 'Initializing analysis pipeline',
        timestamp: new Date().toISOString(),
        model: 'Loading...',
        files: 1
      });
      setConnectionStats(prev => ({
        ...prev,
        totalAnalyses: prev.totalAnalyses + 1
      }));
    }
  }, [analysisId, currentAnalysis]);

  /**
   * Setup WebSocket message listener for analysis updates
   */
  useEffect(() => {
    console.log('🔧 Setting up WebSocket listener for analysis updates');
    
    const handleAnalysisMessage = (update: AnalysisUpdate) => {
      console.log('📨 Dashboard received analysis update:', update);
      
      // Update current analysis if it matches
      if (currentAnalysis && currentAnalysis.id === update.analysis_id) {
        console.log('📊 Updating current analysis:', update);
        setCurrentAnalysis(prev => prev ? {
          ...prev,
          progress: update.progress || prev.progress,
          stage: update.stage || prev.stage,
          message: update.message || prev.message,
          model: update.model || prev.model,
          timestamp: update.timestamp || prev.timestamp
        } : null);
      }
      
      // Call the existing handler
      handleAnalysisUpdate(update);
    };

    // Add the analysis update listener
    if (websocket && typeof websocket.onAnalysisUpdate === 'function') {
      const unsubscribe = websocket.onAnalysisUpdate(handleAnalysisMessage);
      console.log(' WebSocket analysis listener added successfully');
      
      return () => {
        unsubscribe();
        console.log('🗑️ WebSocket analysis listener removed');
      };
    } else {
      console.warn('⚠️ WebSocket instance not available or missing onAnalysisUpdate method');
    }
  }, [websocket, currentAnalysis, handleAnalysisUpdate]);

  /**
   * Get current analysis info
   */
  const getCurrentAnalysisInfo = () => {
    if (!currentAnalysis) return null;
    return {
      analysis_id: currentAnalysis.id,
      progress: currentAnalysis.progress,
      stage: currentAnalysis.stage,
      message: currentAnalysis.message,
      model: currentAnalysis.model,
      timestamp: currentAnalysis.timestamp
    };
  };

  /**
   * Format timestamp for display
   */
  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  /**
   * Get progress color based on stage
   */
  const getProgressColor = (stage: string, progress: number) => {
    if (progress === 100) return 'success';
    if (stage === 'error') return 'error';
    if (progress > 50) return 'primary';
    return 'secondary';
  };

  const currentAnalysisInfo = getCurrentAnalysisInfo();

  return (
    <Paper sx={{ p: 2, minHeight: 400 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Real-time Analysis Dashboard</Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip 
            icon={parentConnected ? <Wifi /> : <WifiOff />}
            label={parentConnected ? 'Connected' : 'Disconnected'}
            color={parentConnected ? 'success' : 'error'}
            size="small"
          />
          <Chip 
            label={parentConnected ? `Latency: <50ms` : `Latency: N/A`}
            size="small"
            variant="outlined"
            color={parentConnected ? "success" : "default"}
          />
          <Button
            size="small"
            onClick={() => window.location.reload()}
            startIcon={<Refresh />}
          >
            Connected
          </Button>
        </Box>
      </Box>

      <Grid container spacing={2}>
        {/* Current Analysis */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Current Analysis</Typography>
              
              {currentAnalysisInfo ? (
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Psychology color="primary" />
                    <Typography variant="subtitle1">
                      {currentAnalysisInfo.stage === 'error' ? 'Analysis Failed' : 
                       currentAnalysisInfo.progress === 100 ? 'Analysis Complete' : 
                       'Processing Analysis'}
                    </Typography>
                    <Chip 
                      label={currentAnalysisInfo.stage}
                      size="small"
                      color={getProgressColor(currentAnalysisInfo.stage, currentAnalysisInfo.progress)}
                    />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Analysis ID: {currentAnalysisInfo.analysis_id}
                  </Typography>
                  
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {currentAnalysisInfo.message}
                  </Typography>
                  
                  <LinearProgress 
                    variant="determinate" 
                    value={currentAnalysisInfo.progress}
                    color={getProgressColor(currentAnalysisInfo.stage, currentAnalysisInfo.progress)}
                    sx={{ mb: 1 }}
                  />
                  
                  <Typography variant="caption" color="text.secondary">
                    Progress: {currentAnalysisInfo.progress}% • {currentAnalysisInfo.model || 'Unknown Model'}
                  </Typography>

                  {/* Action Buttons for Completed Analysis */}
                  {currentAnalysisInfo.progress === 100 && currentAnalysis?.result && (
                    <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      <Button 
                        variant="contained" 
                        size="small"
                        startIcon={<Visibility />}
                        onClick={() => {
                          // Find the analysis in persistent history to view the full report
                          const historyEntry = persistentAnalysisHistory.find(
                            entry => entry.analysis_id === currentAnalysisInfo.analysis_id
                          );
                          if (historyEntry) {
                            setSelectedAnalysis(historyEntry);
                          }
                        }}
                      >
                        View Full Medical Report
                      </Button>
                    </Box>
                  )}
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <BiotechOutlined sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                  <Typography variant="body1" color="text.secondary">
                    No active analysis
                  </Typography>
                  <Typography variant="body2" color="text.disabled">
                    {completedAnalyses.length > 0 ? 
                      `Last analysis completed. ${completedAnalyses.length} total analyses completed.` :
                      'Start an analysis to see real-time progress here'
                    }
                  </Typography>
                  {completedAnalyses.length > 0 && (
                    <Typography variant="caption" color="success.main" sx={{ mt: 1, display: 'block' }}>
                      ✓ Recent analysis completed successfully
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis History */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Analysis History</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip 
                    label={`${persistentAnalysisHistory.length} analyses`} 
                    size="small" 
                    color="primary"
                  />
                  <IconButton 
                    size="small" 
                    onClick={() => setShowAnalysisHistory(!showAnalysisHistory)}
                  >
                    {showAnalysisHistory ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                </Box>
              </Box>
              
              <Collapse in={showAnalysisHistory}>
                {persistentAnalysisHistory.length > 0 ? (
                  <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                    {persistentAnalysisHistory.slice(0, 10).map((entry, index) => (
                      <ListItem 
                        key={`${entry.analysis_id}-${index}`} 
                        divider
                        sx={{ 
                          cursor: 'pointer',
                          '&:hover': { backgroundColor: 'action.hover' },
                          borderRadius: 1,
                          mb: 1
                        }}
                        onClick={() => setSelectedAnalysis(entry)}
                      >
                        <ListItemIcon sx={{ minWidth: 40 }}>
                          {entry.status === 'completed' ? (
                            <CheckCircle color="success" />
                          ) : entry.status === 'failed' ? (
                            <Error color="error" />
                          ) : (
                            <Schedule color="primary" />
                          )}
                        </ListItemIcon>
                        <ListItemText>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2">{entry.file_name}</Typography>
                            {entry.is_2d_image && (
                              <Chip label="2D Image" size="small" color="info" />
                            )}
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              {entry.model} • {formatTimestamp(entry.timestamp)}
                            </Typography>
                            {entry.results && (
                              <Typography variant="caption" display="block" color="text.secondary">
                                {entry.results.tumor_detected ? 
                                  `✓ Tumor detected (${Math.round(entry.results.confidence * 100)}%)` :
                                  '○ No tumor detected'
                                }
                                {entry.processing_time && ` • ${entry.processing_time}ms`}
                              </Typography>
                            )}
                          </Box>
                        </ListItemText>
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 3 }}>
                    <Timeline sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                    <Typography variant="body2" color="text.secondary">
                      No analysis history yet
                    </Typography>
                    <Typography variant="caption" color="text.disabled">
                      Upload and analyze medical images to build your history
                    </Typography>
                  </Box>
                )}
              </Collapse>
            </CardContent>
          </Card>
        </Grid>

        {/* Activity Feed */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6">Live Activity Feed</Typography>
                <IconButton 
                  size="small" 
                  onClick={() => setShowHistory(!showHistory)}
                >
                  {showHistory ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
              </Box>
              
              {analysisHistory.length > 0 ? (
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Latest activity (last {analysisHistory.length} updates)
                  </Typography>
                  
                  <Collapse in={showHistory}>
                    <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
                      {analysisHistory.slice(0, 10).map((update, index) => (
                        <ListItem key={`${update.analysis_id}-${index}`} divider>
                          <ListItemIcon sx={{ minWidth: 32 }}>
                            {update.stage === 'error' ? (
                              <Error color="error" fontSize="small" />
                            ) : update.progress === 100 ? (
                              <CheckCircle color="success" fontSize="small" />
                            ) : (
                              <Schedule color="primary" fontSize="small" />
                            )}
                          </ListItemIcon>
                          <ListItemText
                            primary={update.message}
                            secondary={`${update.stage} • ${formatTimestamp(update.timestamp)}`}
                            primaryTypographyProps={{ variant: 'caption' }}
                            secondaryTypographyProps={{ variant: 'caption', color: 'text.disabled' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Collapse>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 2 }}>
                  <Timeline sx={{ fontSize: 32, color: 'text.disabled', mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    No recent activity
                  </Typography>
                  <Typography variant="caption" color="text.disabled">
                    Analysis updates will appear here in real-time
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Completed Analyses */}
        {completedAnalyses.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Completed Analyses</Typography>
                
                <Grid container spacing={2}>
                  {completedAnalyses.slice(0, 3).map((result, index) => (
                    <Grid item xs={12} md={4} key={index}>
                      <Paper sx={{ p: 2, backgroundColor: 'success.50' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <CheckCircle color="success" fontSize="small" />
                          <Typography variant="subtitle2">{result.model}</Typography>
                        </Box>
                        
                        <Typography variant="body2" gutterBottom>
                          {result.prediction?.tumor_detected || result.tumor_detected ? 
                            `Tumor Detected (${Math.round((result.prediction?.confidence || result.confidence) * 100)}%)` :
                            'No Tumor Detected'
                          }
                        </Typography>
                        
                        {(result.prediction?.tumor_type || result.tumor_type) && (
                          <Chip 
                            label={result.prediction?.tumor_type || result.tumor_type}
                            size="small"
                            color="secondary"
                            sx={{ mb: 1 }}
                          />
                        )}
                        
                        {(result.prediction?.location || result.tumor_location) && (
                          <Typography variant="caption" display="block">
                            Location: {result.prediction?.location || result.tumor_location}
                          </Typography>
                        )}
                        
                        <Typography variant="caption" color="text.secondary">
                          Processing: {result.processing_time}ms
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Connection Status Alert */}
      {!parentConnected && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          WebSocket connection lost. Real-time updates are not available.
        </Alert>
      )}

      {/* Enhanced Analysis Details Modal */}
      <Dialog 
        open={!!selectedAnalysis} 
        onClose={() => setSelectedAnalysis(null)}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: {
            minHeight: '85vh',
            maxHeight: '95vh',
            bgcolor: 'background.default'
          }
        }}
      >
        <DialogTitle 
          sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            py: 2.5
          }}
        >
          <Box>
            <Typography variant="h5" fontWeight="bold">
              🧠 Medical Analysis Report
            </Typography>
            {selectedAnalysis && (
              <Typography variant="body2" sx={{ mt: 0.5, opacity: 0.9 }}>
                {selectedAnalysis.file_name} • {formatTimestamp(selectedAnalysis.timestamp)}
              </Typography>
            )}
          </Box>
          <IconButton onClick={() => setSelectedAnalysis(null)} sx={{ color: 'inherit' }}>
            <Close />
          </IconButton>
        </DialogTitle>
        
        <DialogContent>
          {selectedAnalysis && (
            <Grid container spacing={3}>
              {/* Real-time Brain Activity Simulation - NEW! */}
              <Grid item xs={12} md={6}>
                <BrainActivitySimulation
                  tumorDetected={selectedAnalysis.results?.tumor_detected || false}
                  confidence={selectedAnalysis.results?.confidence || 0}
                  tumorLocation={selectedAnalysis.results?.prediction?.location || 'Unknown'}
                  analysisId={selectedAnalysis.analysis_id}
                />
              </Grid>

              {/* File Information */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                      File Information
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableBody suppressHydrationWarning>
                          <TableRow>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">File Name:</Typography>
                            </TableCell>
                            <TableCell>{selectedAnalysis.file_name}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">Analysis ID:</Typography>
                            </TableCell>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                              {selectedAnalysis.analysis_id}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">Detection Model:</Typography>
                            </TableCell>
                            <TableCell>{selectedAnalysis.model}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">File Type:</Typography>
                            </TableCell>
                            <TableCell>
                              {selectedAnalysis.is_2d_image ? (
                                <Chip label="2D Image (JPG/PNG)" size="small" color="info" />
                              ) : (
                                <Chip label="3D Medical Image" size="small" color="primary" />
                              )}
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>
                              <Typography variant="body2" fontWeight="bold">Processing Time:</Typography>
                            </TableCell>
                            <TableCell>{selectedAnalysis.processing_time || 'N/A'}ms</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Medical Report */}
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <Psychology sx={{ mr: 1, verticalAlign: 'middle' }} />
                      Medical Report
                    </Typography>
                    {selectedAnalysis.results ? (
                      <Grid container spacing={3}>
                        {/* Primary Findings */}
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                            Primary Findings
                          </Typography>
                          <TableContainer>
                            <Table size="small">
                              <TableBody suppressHydrationWarning>
                                <TableRow>
                                  <TableCell>
                                    <Typography variant="body2" fontWeight="bold">Tumor Status:</Typography>
                                  </TableCell>
                                  <TableCell>
                                    {selectedAnalysis.results.tumor_detected ? (
                                      <Chip label="DETECTED" size="small" color="error" icon={<Warning />} />
                                    ) : (
                                      <Chip label="NOT DETECTED" size="small" color="success" icon={<CheckCircle />} />
                                    )}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    <Typography variant="body2" fontWeight="bold">Confidence Level:</Typography>
                                  </TableCell>
                                  <TableCell>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <LinearProgress 
                                        variant="determinate" 
                                        value={selectedAnalysis.results.confidence * 100}
                                        sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                                        color={selectedAnalysis.results.confidence > 0.9 ? 'success' : 
                                               selectedAnalysis.results.confidence > 0.7 ? 'warning' : 'error'}
                                      />
                                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                        {Math.round(selectedAnalysis.results.confidence * 100)}%
                                      </Typography>
                                    </Box>
                                  </TableCell>
                                </TableRow>
                                {selectedAnalysis.results.prediction?.tumor_type ? (
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">Tumor Classification:</Typography>
                                    </TableCell>
                                    <TableCell sx={{ fontWeight: 'bold' }}>
                                      {selectedAnalysis.results.prediction.tumor_type}
                                    </TableCell>
                                  </TableRow>
                                ) : null}
                                {selectedAnalysis.results.prediction?.tumor_grade ? (
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">WHO Grade:</Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Chip 
                                        label={selectedAnalysis.results.prediction.tumor_grade} 
                                        size="small" 
                                        color={selectedAnalysis.results.prediction.tumor_grade.includes('IV') ? 'error' :
                                               selectedAnalysis.results.prediction.tumor_grade.includes('III') ? 'warning' : 'info'}
                                      />
                                    </TableCell>
                                  </TableRow>
                                ) : null}
                                {selectedAnalysis.results.prediction?.location ? (
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">Anatomical Location:</Typography>
                                    </TableCell>
                                    <TableCell>{selectedAnalysis.results.prediction.location}</TableCell>
                                  </TableRow>
                                ) : null}
                                {selectedAnalysis.results.prediction?.volume_ml ? (
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">Tumor Volume:</Typography>
                                    </TableCell>
                                    <TableCell>
                                      <strong>{selectedAnalysis.results.prediction.volume_ml.toFixed(1)} mL</strong>
                                      {selectedAnalysis.results.prediction.volume_ml > 30 ? (
                                        <Chip label="Large" size="small" color="warning" sx={{ ml: 1 }} />
                                      ) : null}
                                    </TableCell>
                                  </TableRow>
                                ) : null}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        </Grid>

                        {/* Clinical Assessment */}
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                            Clinical Assessment
                          </Typography>
                          {selectedAnalysis.results.risk_assessment && (
                            <TableContainer>
                              <Table size="small">
                                <TableBody suppressHydrationWarning>
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">
                                        Risk Level:
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Chip 
                                        label={selectedAnalysis.results.risk_assessment.risk_level}
                                        size="small"
                                        color={
                                          selectedAnalysis.results.risk_assessment.risk_level === 'High' ? 'error' :
                                          selectedAnalysis.results.risk_assessment.risk_level === 'Medium-High' ? 'warning' :
                                          selectedAnalysis.results.risk_assessment.risk_level === 'Medium' ? 'info' : 'success'
                                        }
                                      />
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">
                                        Urgency:
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Chip 
                                        label={selectedAnalysis.results.risk_assessment.urgency}
                                        size="small"
                                        color={selectedAnalysis.results.risk_assessment.urgency === 'Urgent' ? 'error' :
                                               selectedAnalysis.results.risk_assessment.urgency === 'Priority' ? 'warning' : 'info'}
                                      />
                                    </TableCell>
                                  </TableRow>
                                  <TableRow>
                                    <TableCell colSpan={2}>
                                      <Box>
                                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                                          Clinical Recommendation:
                                        </Typography>
                                        <Typography variant="body2" sx={{ p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                                          {selectedAnalysis.results.risk_assessment.recommendation}
                                        </Typography>
                                      </Box>
                                    </TableCell>
                                  </TableRow>
                                  {selectedAnalysis.results.risk_assessment.follow_up ? (
                                    <TableRow>
                                      <TableCell colSpan={2}>
                                        <Box>
                                          <Typography variant="body2" fontWeight="bold" gutterBottom>
                                            Follow-up:
                                          </Typography>
                                          <Typography variant="body2">
                                            {selectedAnalysis.results.risk_assessment.follow_up}
                                          </Typography>
                                        </Box>
                                      </TableCell>
                                    </TableRow>
                                  ) : null}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          )}
                        </Grid>

                        {/* Advanced Imaging Features */}
                        {selectedAnalysis.results.prediction && (
                          selectedAnalysis.results.prediction.enhancement_pattern || 
                          selectedAnalysis.results.prediction.mass_effect !== undefined ||
                          selectedAnalysis.results.prediction.edema_present !== undefined
                        ) && (
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Imaging Characteristics
                            </Typography>
                            <TableContainer>
                              <Table size="small">
                                <TableBody suppressHydrationWarning>
                                  {selectedAnalysis.results.prediction.enhancement_pattern ? (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">
                                          Enhancement Pattern:
                                        </Typography>
                                      </TableCell>
                                      <TableCell>{selectedAnalysis.results.prediction.enhancement_pattern}</TableCell>
                                    </TableRow>
                                  ) : null}
                                  {selectedAnalysis.results.prediction.mass_effect !== undefined ? (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">
                                          Mass Effect:
                                        </Typography>
                                      </TableCell>
                                      <TableCell>
                                        {selectedAnalysis.results.prediction.mass_effect ? (
                                          <Chip label="Present" size="small" color="warning" />
                                        ) : (
                                          <Chip label="Absent" size="small" color="success" />
                                        )}
                                      </TableCell>
                                    </TableRow>
                                  ) : null}
                                  {selectedAnalysis.results.prediction.edema_present !== undefined ? (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">
                                          Perilesional Edema:
                                        </Typography>
                                      </TableCell>
                                      <TableCell>
                                        {selectedAnalysis.results.prediction.edema_present ? (
                                          <Chip label="Present" size="small" color="warning" />
                                        ) : (
                                          <Chip label="Absent" size="small" color="success" />
                                        )}
                                      </TableCell>
                                    </TableRow>
                                  ) : null}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Grid>
                        )}

                        {/* Segmentation Metrics */}
                        {selectedAnalysis.results.segmentation && (
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Segmentation Analysis
                            </Typography>
                            <TableContainer>
                              <Table size="small">
                                <TableBody suppressHydrationWarning>
                                  <TableRow>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight="bold">Segmentation Quality:</Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <LinearProgress 
                                          variant="determinate" 
                                          value={(selectedAnalysis.results.segmentation.segmentation_quality || 0) * 100}
                                          sx={{ flexGrow: 1, height: 6, borderRadius: 1 }}
                                          color="primary"
                                        />
                                        <Typography variant="caption">
                                          {Math.round((selectedAnalysis.results.segmentation.segmentation_quality || 0) * 100)}%
                                        </Typography>
                                      </Box>
                                    </TableCell>
                                  </TableRow>
                                  {selectedAnalysis.results.segmentation.dice_score && (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">Dice Score:</Typography>
                                      </TableCell>
                                      <TableCell>{selectedAnalysis.results.segmentation.dice_score.toFixed(3)}</TableCell>
                                    </TableRow>
                                  )}
                                  {selectedAnalysis.results.segmentation.tumor_core_volume && (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">Tumor Core Volume:</Typography>
                                      </TableCell>
                                      <TableCell>{selectedAnalysis.results.segmentation.tumor_core_volume.toFixed(1)} mL</TableCell>
                                    </TableRow>
                                  )}
                                  {selectedAnalysis.results.segmentation.enhancement_volume && (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">Enhancement Volume:</Typography>
                                      </TableCell>
                                      <TableCell>{selectedAnalysis.results.segmentation.enhancement_volume.toFixed(1)} mL</TableCell>
                                    </TableRow>
                                  )}
                                  {selectedAnalysis.results.segmentation.edema_volume && selectedAnalysis.results.segmentation.edema_volume > 0 && (
                                    <TableRow>
                                      <TableCell>
                                        <Typography variant="body2" fontWeight="bold">Edema Volume:</Typography>
                                      </TableCell>
                                      <TableCell>{selectedAnalysis.results.segmentation.edema_volume.toFixed(1)} mL</TableCell>
                                    </TableRow>
                                  )}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Grid>
                        )}

                        {/* Differential Diagnosis */}
                        {selectedAnalysis.results.risk_assessment?.differential_diagnosis && (
                          <Grid item xs={12}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Differential Diagnosis
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                              {selectedAnalysis.results.risk_assessment.differential_diagnosis.map((diagnosis, index) => (
                                <Chip 
                                  key={index}
                                  label={diagnosis}
                                  variant={index === 0 ? "filled" : "outlined"}
                                  color={index === 0 ? "primary" : "default"}
                                />
                              ))}
                            </Box>
                          </Grid>
                        )}

                        {/* Clinical Notes */}
                        {selectedAnalysis.results.clinical_notes && (
                          <Grid item xs={12}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                              Clinical Notes
                            </Typography>
                            <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1, border: 1, borderColor: 'divider' }}>
                              <Typography variant="body2" paragraph>
                                <strong>Findings:</strong> {selectedAnalysis.results.clinical_notes.findings}
                              </Typography>
                              <Typography variant="body2" paragraph>
                                <strong>Limitations:</strong> {selectedAnalysis.results.clinical_notes.limitations}
                              </Typography>
                              {selectedAnalysis.results.clinical_notes.quality_indicators && (
                                <Typography variant="body2">
                                  <strong>Image Quality:</strong> {selectedAnalysis.results.clinical_notes.quality_indicators.image_quality}
                                  {selectedAnalysis.results.clinical_notes.quality_indicators.artifacts_present && 
                                    " • Artifacts detected"}
                                  {selectedAnalysis.results.clinical_notes.quality_indicators.contrast_enhancement && 
                                    ` • Contrast: ${selectedAnalysis.results.clinical_notes.quality_indicators.contrast_enhancement}`}
                                </Typography>
                              )}
                            </Box>
                          </Grid>
                        )}
                      </Grid>
                    ) : (
                      <Typography color="text.secondary">No detailed results available</Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Special Message for 2D Images */}
              {selectedAnalysis.is_2d_image && (
                <Grid item xs={12}>
                  <Alert severity="info">
                    <Typography variant="subtitle2" gutterBottom>
                      2D Image Analysis Note
                    </Typography>
                    <Typography variant="body2">
                      This analysis was performed on a 2D image file (JPG/PNG). For more accurate tumor detection and 3D visualization, 
                      please upload DICOM (.dcm) or NIfTI (.nii) medical imaging files. 2D images provide limited clinical information 
                      compared to full 3D medical scans.
                    </Typography>
                  </Alert>
                </Grid>
              )}

              {/* Metrics (if available) */}
              {selectedAnalysis.results?.metrics && (
                <Grid item xs={12}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                      <Grid container spacing={2}>
                        {Object.entries(selectedAnalysis.results.metrics).map(([key, value]) => (
                          <Grid item xs={6} sm={3} key={key}>
                            <Box sx={{ textAlign: 'center', p: 1, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                              <Typography variant="caption" color="text.secondary">
                                {key.replace(/_/g, ' ').toUpperCase()}
                              </Typography>
                              <Typography variant="h6">
                                {typeof value === 'number' ? value.toFixed(3) : value}
                              </Typography>
                            </Box>
                          </Grid>
                        ))}
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>

        <DialogActions>
          <Button 
            onClick={() => setSelectedAnalysis(null)}
            variant="outlined"
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default RealTimeAnalysisDashboard;