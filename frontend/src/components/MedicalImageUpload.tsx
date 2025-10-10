/**
 * Medical Image Upload Component with Real-time Analysis
 * Handles file upload and triggers real-time AI analysis
 */

'use client';

import React, { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  CloudUpload,
  Image as ImageIcon,
  Delete,
  Visibility,
  Analytics,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Psychology,
  Speed,
  ModelTraining
} from '@mui/icons-material';
import { useWebSocket } from '../lib/websocket';
import { toast } from 'react-hot-toast';

interface UploadedFile {
  file: File;
  id: string;
  preview?: string;
  status: 'pending' | 'uploading' | 'analyzing' | 'complete' | 'error';
  progress: number;
  analysisId?: string;
  error?: string;
}

interface ModelInfo {
  id: string;
  name: string;
  description: string;
  type: string;
  estimatedTime: string;
}

interface MedicalImageUploadProps {
  onAnalysisStart?: (analysisId: string, files: UploadedFile[], model: string) => void;
  onAnalysisComplete?: (results: any) => void;
  onFilePreview?: (file: File) => void;
}

// Available AI models
const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'ensemble',
    name: 'Advanced Ensemble Model',
    description: 'Combines multiple AI models for superior accuracy',
    type: 'ensemble',
    estimatedTime: '3-5 minutes'
  },
  {
    id: 'nnunet',
    name: 'nnU-Net Segmentation',
    description: 'State-of-the-art medical image segmentation',
    type: 'segmentation',
    estimatedTime: '2-3 minutes'
  },
  {
    id: 'medvit',
    name: 'Medical Vision Transformer',
    description: 'Advanced transformer architecture for medical imaging',
    type: 'classification',
    estimatedTime: '1-2 minutes'
  },
  {
    id: 'unet3d',
    name: '3D U-Net',
    description: 'Volumetric tumor segmentation',
    type: 'segmentation',
    estimatedTime: '2 minutes'
  },
  {
    id: 'resnet3d',
    name: '3D ResNet Classifier',
    description: 'Deep residual network for tumor classification',
    type: 'classification',
    estimatedTime: '30 seconds'
  },
  {
    id: 'multimodal',
    name: 'Multi-Modal CNN',
    description: 'Combined analysis of multiple image sequences',
    type: 'multimodal',
    estimatedTime: '3-4 minutes'
  }
];

export const MedicalImageUpload: React.FC<MedicalImageUploadProps> = ({
  onAnalysisStart,
  onAnalysisComplete,
  onFilePreview
}) => {
  const websocket = useWebSocket();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [previewDialog, setPreviewDialog] = useState<{ open: boolean; file?: File }>({ open: false });
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Handle file drop/selection
   */
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      file,
      id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
      status: 'pending',
      progress: 0
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
    toast.success(`${acceptedFiles.length} file(s) uploaded successfully`);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'],
      'application/dicom': ['.dcm', '.dicom'],
      'application/octet-stream': ['.nii', '.nii.gz']
    },
    maxFiles: 10,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  /**
   * Upload file to backend
   */
  const uploadFile = async (uploadedFile: UploadedFile): Promise<string | null> => {
    const formData = new FormData();
    formData.append('files', uploadedFile.file); // Backend expects 'files'
    formData.append('model', selectedModel);

    try {
      setUploadedFiles(prev =>
        prev.map(f => f.id === uploadedFile.id ? { ...f, status: 'uploading', progress: 0 } : f)
      );

      const response = await fetch('http://localhost:8000/api/v1/analysis/upload', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer mock-token', // Add demo authentication
        },
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      setUploadedFiles(prev =>
        prev.map(f => f.id === uploadedFile.id ? { 
          ...f, 
          status: 'analyzing', 
          progress: 10, 
          analysisId: data.analysis_id 
        } : f)
      );

      return data.analysis_id;
    } catch (error: any) {
      console.error('Upload error:', error);
      setUploadedFiles(prev =>
        prev.map(f => f.id === uploadedFile.id ? { 
          ...f, 
          status: 'error', 
          error: error instanceof Error ? error.message : 'Upload failed' 
        } : f)
      );
      toast.error(error instanceof Error ? error.message : 'Upload failed');
      return null;
    }
  };

  /**
   * Start analysis for a specific file
   */
  const startAnalysis = async (uploadedFile: UploadedFile) => {
    const analysisId = await uploadFile(uploadedFile);
    if (analysisId && onAnalysisStart) {
      onAnalysisStart(analysisId, [uploadedFile], selectedModel);
    }
    return analysisId;
  };

  /**
   * Start analysis for all files
   */
  const startBatchAnalysis = async () => {
    if (uploadedFiles.length === 0) {
      toast.error('Please upload files first');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
      let firstAnalysisId: string | null = null;
      
      for (const file of pendingFiles) {
        const analysisId = await startAnalysis(file);
        if (!firstAnalysisId && analysisId) {
          firstAnalysisId = analysisId;
        }
      }
      
      toast.success('Analysis started for all files');
    } catch (error) {
      console.error('Batch analysis error:', error);
      toast.error('Failed to start batch analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  /**
   * Remove file from upload list
   */
  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => {
      const file = prev.find(f => f.id === fileId);
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
      return prev.filter(f => f.id !== fileId);
    });
    toast.success('File removed');
  };

  /**
   * Clear all files
   */
  const clearAllFiles = () => {
    uploadedFiles.forEach(file => {
      if (file.preview) {
        URL.revokeObjectURL(file.preview);
      }
    });
    setUploadedFiles([]);
    toast.success('All files cleared');
  };

  /**
   * Preview file
   */
  const previewFile = (file: File) => {
    setPreviewDialog({ open: true, file });
    if (onFilePreview) {
      onFilePreview(file);
    }
  };

  /**
   * Get status icon and color for file
   */
  const getFileStatus = (file: UploadedFile) => {
    switch (file.status) {
      case 'pending':
        return { color: 'default' as const, icon: <CloudUpload /> };
      case 'uploading':
        return { color: 'primary' as const, icon: <CloudUpload /> };
      case 'analyzing':
        return { color: 'secondary' as const, icon: <Analytics /> };
      case 'complete':
        return { color: 'success' as const, icon: <CheckCircle /> };
      case 'error':
        return { color: 'error' as const, icon: <ErrorIcon /> };
      default:
        return { color: 'default' as const, icon: <CloudUpload /> };
    }
  };

  /**
   * Get model icon based on type
   */
  const getModelIcon = (type: string) => {
    switch (type) {
      case 'ensemble':
        return <Psychology />;
      case 'segmentation':
        return <Analytics />;
      case 'classification':
        return <Speed />;
      case 'multimodal':
        return <ModelTraining />;
      default:
        return <Analytics />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
        <CloudUpload sx={{ mr: 2 }} />
        Medical Image Upload & Analysis
      </Typography>

      {/* Upload Area */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          mb: 3,
          border: 2,
          borderStyle: 'dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          bgcolor: isDragActive ? 'primary.50' : 'grey.50',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'primary.50'
          }
        }}
      >
        <input {...getInputProps()} />
        <Box textAlign="center">
          <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive
              ? 'Drop medical images here...'
              : 'Drag & drop medical images or click to select'
            }
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Supports: DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG, JPEG, TIFF
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Maximum file size: 100MB | Maximum files: 10
          </Typography>
        </Box>
      </Paper>

      {/* Model Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            AI Model Selection
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Select AI Model</InputLabel>
            <Select
              value={selectedModel}
              label="Select AI Model"
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {AVAILABLE_MODELS.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {getModelIcon(model.type)}
                    <Box sx={{ ml: 2 }}>
                      <Typography variant="body1">{model.name}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {model.description} • {model.estimatedTime}
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {/* Selected Model Info */}
          {selectedModel && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                <strong>{AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}</strong>
                <br />
                {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.description}
                <br />
                Estimated processing time: {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.estimatedTime}
              </Typography>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Uploaded Files ({uploadedFiles.length})
              </Typography>
              <Box>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={startBatchAnalysis}
                  disabled={isAnalyzing || uploadedFiles.every(f => f.status !== 'pending')}
                  sx={{ mr: 1 }}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={clearAllFiles}
                  disabled={isAnalyzing}
                >
                  Clear All
                </Button>
              </Box>
            </Box>

            <List>
              {uploadedFiles.map((uploadedFile) => {
                const status = getFileStatus(uploadedFile);
                return (
                  <ListItem key={uploadedFile.id} divider>
                    <ListItemIcon>
                      <Chip
                        icon={status.icon}
                        label={uploadedFile.status}
                        color={status.color}
                        size="small"
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={uploadedFile.file.name}
                      secondary={
                        <React.Fragment>
                          <span style={{ fontSize: '0.875rem', color: 'rgba(0, 0, 0, 0.6)' }}>
                            Size: {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB
                            {uploadedFile.error && (
                              <span style={{ color: '#d32f2f', marginLeft: '8px' }}>
                                • {uploadedFile.error}
                              </span>
                            )}
                          </span>
                          {uploadedFile.status === 'analyzing' && (
                            <LinearProgress
                              variant="determinate"
                              value={uploadedFile.progress}
                              sx={{ mt: 1, display: 'block' }}
                            />
                          )}
                        </React.Fragment>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => previewFile(uploadedFile.file)}
                        disabled={!uploadedFile.file.type.startsWith('image/')}
                        sx={{ mr: 1 }}
                      >
                        <Visibility />
                      </IconButton>
                      <IconButton
                        edge="end"
                        onClick={() => removeFile(uploadedFile.id)}
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                );
              })}
            </List>
          </CardContent>
        </Card>
      )}

      {/* File Preview Dialog */}
      <Dialog
        open={previewDialog.open}
        onClose={() => setPreviewDialog({ open: false })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>File Preview</DialogTitle>
        <DialogContent>
          {previewDialog.file && previewDialog.file.type.startsWith('image/') && (
            <Box sx={{ textAlign: 'center' }}>
              <img
                src={URL.createObjectURL(previewDialog.file)}
                alt="File preview"
                style={{ maxWidth: '100%', maxHeight: '400px' }}
              />
              <Typography variant="body2" sx={{ mt: 2 }}>
                {previewDialog.file.name} - {(previewDialog.file.size / 1024 / 1024).toFixed(2)} MB
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialog({ open: false })}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Usage Instructions */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            How to Use
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <CloudUpload sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6">1. Upload</Typography>
                <Typography variant="body2">
                  Drag & drop or click to select medical images
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <Psychology sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6">2. Select Model</Typography>
                <Typography variant="body2">
                  Choose an AI model based on your analysis needs
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center', p: 2 }}>
                <Analytics sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                <Typography variant="h6">3. Analyze</Typography>
                <Typography variant="body2">
                  Start analysis and monitor real-time progress
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MedicalImageUpload;