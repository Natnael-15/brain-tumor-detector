'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Divider,
  LinearProgress,
  Alert,
  Button,
  List,
  ListItem,
  ListItemText,
  Grid,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';

interface AnalysisResultsProps {
  analysisId: string;
}

interface AnalysisResult {
  analysis_id: string;
  model_used: string;
  predictions: {
    tumor_detected: boolean;
    tumor_type: string;
    confidence: number;
    tumor_volume_ml: number;
    location: string;
  };
  metrics: {
    dice_score: number;
    hausdorff_distance: number;
    processing_time: number;
  };
  clinical_notes: string[];
  completed_at: string;
}

export function AnalysisResults({ analysisId }: AnalysisResultsProps) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Mock data loading
    const loadResults = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock result data
        const mockResult: AnalysisResult = {
          analysis_id: analysisId,
          model_used: 'ensemble',
          predictions: {
            tumor_detected: true,
            tumor_type: 'Glioblastoma',
            confidence: 0.87,
            tumor_volume_ml: 12.5,
            location: 'Right frontal lobe',
          },
          metrics: {
            dice_score: 0.92,
            hausdorff_distance: 2.1,
            processing_time: 4.2,
          },
          clinical_notes: [
            'Enhancing lesion identified in right frontal lobe',
            'Irregular borders suggestive of high-grade glioma',
            'Recommend correlation with clinical symptoms',
            'Consider follow-up imaging in 3 months',
          ],
          completed_at: new Date().toISOString(),
        };
        
        setResult(mockResult);
        setLoading(false);
      } catch (err) {
        setError('Failed to load analysis results');
        setLoading(false);
      }
    };

    loadResults();
  }, [analysisId]);

  if (loading) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Loading Results...
          </Typography>
          <LinearProgress sx={{ mt: 2 }} />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Alert severity="error" icon={<ErrorIcon />}>
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!result) {
    return null;
  }

  const { predictions, metrics, clinical_notes } = result;

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom display="flex" alignItems="center">
          <CheckIcon color="success" sx={{ mr: 1 }} />
          Analysis Complete
        </Typography>
        
        <Typography variant="body2" color="text.secondary" mb={2}>
          Model: {result.model_used} • Completed: {new Date(result.completed_at).toLocaleString()}
        </Typography>

        <Divider sx={{ my: 2 }} />

        {/* Tumor Detection */}
        <Box mb={3}>
          <Typography variant="subtitle1" fontWeight={500} gutterBottom>
            Detection Results
          </Typography>
          
          <Alert
            severity={predictions.tumor_detected ? 'warning' : 'success'}
            icon={predictions.tumor_detected ? <WarningIcon /> : <CheckIcon />}
            sx={{ mb: 2 }}
          >
            {predictions.tumor_detected
              ? `Tumor detected: ${predictions.tumor_type}`
              : 'No tumor detected'}
          </Alert>

          {predictions.tumor_detected && (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Confidence
                </Typography>
                <Typography variant="h6" color="primary">
                  {(predictions.confidence * 100).toFixed(1)}%
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Volume
                </Typography>
                <Typography variant="h6">
                  {predictions.tumor_volume_ml} mL
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  Location
                </Typography>
                <Typography variant="body1">
                  {predictions.location}
                </Typography>
              </Grid>
            </Grid>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Metrics */}
        <Box mb={3}>
          <Typography variant="subtitle1" fontWeight={500} gutterBottom>
            Quality Metrics
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">
                Dice Score
              </Typography>
              <Typography variant="h6">
                {metrics.dice_score.toFixed(2)}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">
                Hausdorff (mm)
              </Typography>
              <Typography variant="h6">
                {metrics.hausdorff_distance.toFixed(1)}
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" color="text.secondary">
                Time (s)
              </Typography>
              <Typography variant="h6">
                {metrics.processing_time.toFixed(1)}
              </Typography>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Clinical Notes */}
        <Box mb={3}>
          <Typography variant="subtitle1" fontWeight={500} gutterBottom>
            Clinical Notes
          </Typography>
          
          <List dense>
            {clinical_notes.map((note, index) => (
              <ListItem key={index} sx={{ pl: 0 }}>
                <ListItemText
                  primary={note}
                  primaryTypographyProps={{ variant: 'body2' }}
                />
              </ListItem>
            ))}
          </List>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Actions */}
        <Box display="flex" gap={1} flexWrap="wrap">
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            size="small"
          >
            Download Report
          </Button>
          <Button
            variant="outlined"
            size="small"
          >
            View 3D
          </Button>
          <Button
            variant="outlined"
            size="small"
          >
            DICOM Export
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
}