/**
 * Real-time Brain Activity Simulation Component
 * Displays animated brain activity visualization for medical reports
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Box, Paper, Typography, Chip, Grid, LinearProgress } from '@mui/material';
import { Psychology, Warning, CheckCircle, Healing } from '@mui/icons-material';

interface BrainActivitySimulationProps {
  tumorDetected?: boolean;
  confidence?: number;
  tumorLocation?: string;
  analysisId?: string;
}

export const BrainActivitySimulation: React.FC<BrainActivitySimulationProps> = ({
  tumorDetected = false,
  confidence = 0,
  tumorLocation = 'Unknown',
  analysisId
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [brainWaves, setBrainWaves] = useState({
    alpha: 65,
    beta: 48,
    gamma: 72,
    delta: 55
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;
    const animate = () => {
      const width = canvas.width;
      const height = canvas.height;

      // Clear canvas with fade effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, width, height);

      // Draw brain outline
      drawBrainOutline(ctx, width, height);

      // Draw neural activity
      drawNeuralActivity(ctx, width, height, time, tumorDetected);

      // Draw tumor location if detected
      if (tumorDetected) {
        drawTumorIndicator(ctx, width, height, time);
      }

      time += 0.02;

      // Update brain wave readings periodically
      if (Math.floor(time * 10) % 20 === 0) {
        setBrainWaves({
          alpha: 60 + Math.random() * 20,
          beta: 40 + Math.random() * 25,
          gamma: 65 + Math.random() * 20,
          delta: 50 + Math.random() * 15
        });
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    // Start animation
    animate();

    // Cleanup
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [tumorDetected]);

  const drawBrainOutline = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const radiusX = width * 0.35;
    const radiusY = height * 0.4;

    ctx.strokeStyle = 'rgba(100, 200, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw brain hemispheres division
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - radiusY);
    ctx.lineTo(centerX, centerY + radiusY);
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
  };

  const drawNeuralActivity = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    time: number,
    hasTumor: boolean
  ) => {
    const centerX = width / 2;
    const centerY = height / 2;

    // Draw multiple activity nodes
    for (let i = 0; i < 12; i++) {
      const angle = (i / 12) * Math.PI * 2 + time;
      const radius = 80 + Math.sin(time * 2 + i) * 20;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      // Node glow
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
      if (hasTumor && i === 3) {
        // Highlight tumor area
        gradient.addColorStop(0, 'rgba(255, 80, 80, 0.8)');
        gradient.addColorStop(1, 'rgba(255, 80, 80, 0)');
      } else {
        gradient.addColorStop(0, 'rgba(80, 200, 255, 0.6)');
        gradient.addColorStop(1, 'rgba(80, 200, 255, 0)');
      }

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 15, 0, Math.PI * 2);
      ctx.fill();

      // Draw connections between nodes
      if (i < 11) {
        const nextAngle = ((i + 1) / 12) * Math.PI * 2 + time;
        const nextRadius = 80 + Math.sin(time * 2 + i + 1) * 20;
        const nextX = centerX + Math.cos(nextAngle) * nextRadius;
        const nextY = centerY + Math.sin(nextAngle) * nextRadius;

        ctx.strokeStyle = hasTumor && (i === 2 || i === 3) 
          ? 'rgba(255, 100, 100, 0.3)' 
          : 'rgba(80, 200, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(nextX, nextY);
        ctx.stroke();
      }
    }

    // Pulsing center activity
    const pulseRadius = 30 + Math.sin(time * 3) * 10;
    const centerGradient = ctx.createRadialGradient(
      centerX, centerY, 0, 
      centerX, centerY, pulseRadius
    );
    centerGradient.addColorStop(0, hasTumor ? 'rgba(255, 150, 80, 0.5)' : 'rgba(100, 255, 200, 0.5)');
    centerGradient.addColorStop(1, 'rgba(100, 200, 255, 0)');
    ctx.fillStyle = centerGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
    ctx.fill();
  };

  const drawTumorIndicator = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    time: number
  ) => {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Draw tumor marker (right hemisphere, frontal region)
    const tumorX = centerX + 60;
    const tumorY = centerY - 40;
    const pulse = Math.sin(time * 4) * 5;

    // Warning indicator
    ctx.fillStyle = 'rgba(255, 50, 50, 0.7)';
    ctx.beginPath();
    ctx.arc(tumorX, tumorY, 12 + pulse, 0, Math.PI * 2);
    ctx.fill();

    // Outer warning ring
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tumorX, tumorY, 20 + pulse * 2, 0, Math.PI * 2);
    ctx.stroke();
  };

  return (
    <Paper elevation={3} sx={{ p: 2, bgcolor: 'background.paper', height: '100%' }}>
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Psychology color="primary" sx={{ fontSize: 28 }} />
          <Typography variant="h6" fontWeight="bold">
            Real-time Brain Activity Monitor
          </Typography>
        </Box>
        <Chip 
          icon={tumorDetected ? <Warning /> : <CheckCircle />}
          label={tumorDetected ? "Anomaly Detected" : "Normal Activity"}
          color={tumorDetected ? "error" : "success"}
          size="small"
        />
      </Box>

      {/* Canvas for brain visualization */}
      <Box 
        sx={{ 
          position: 'relative',
          width: '100%',
          height: 300,
          bgcolor: 'rgba(0, 0, 0, 0.9)',
          borderRadius: 2,
          overflow: 'hidden',
          mb: 2
        }}
      >
        <canvas
          ref={canvasRef}
          width={600}
          height={300}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
      </Box>

      {/* Brain Wave Readings */}
      <Grid container spacing={2}>
        <Grid item xs={6} sm={3}>
          <Box>
            <Typography variant="caption" color="text.secondary">Alpha Waves</Typography>
            <LinearProgress 
              variant="determinate" 
              value={brainWaves.alpha} 
              sx={{ height: 8, borderRadius: 1, my: 0.5 }}
              color="info"
            />
            <Typography variant="body2" fontWeight="bold">{brainWaves.alpha.toFixed(0)}%</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Box>
            <Typography variant="caption" color="text.secondary">Beta Waves</Typography>
            <LinearProgress 
              variant="determinate" 
              value={brainWaves.beta} 
              sx={{ height: 8, borderRadius: 1, my: 0.5 }}
              color="success"
            />
            <Typography variant="body2" fontWeight="bold">{brainWaves.beta.toFixed(0)}%</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Box>
            <Typography variant="caption" color="text.secondary">Gamma Waves</Typography>
            <LinearProgress 
              variant="determinate" 
              value={brainWaves.gamma} 
              sx={{ height: 8, borderRadius: 1, my: 0.5 }}
              color="warning"
            />
            <Typography variant="body2" fontWeight="bold">{brainWaves.gamma.toFixed(0)}%</Typography>
          </Box>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Box>
            <Typography variant="caption" color="text.secondary">Delta Waves</Typography>
            <LinearProgress 
              variant="determinate" 
              value={brainWaves.delta} 
              sx={{ height: 8, borderRadius: 1, my: 0.5 }}
              color="error"
            />
            <Typography variant="body2" fontWeight="bold">{brainWaves.delta.toFixed(0)}%</Typography>
          </Box>
        </Grid>
      </Grid>

      {tumorDetected && (
        <Box sx={{ mt: 2, p: 1.5, bgcolor: 'error.light', color: 'error.dark', borderRadius: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Healing />
            <Typography variant="body2" fontWeight="bold">
              Clinical Alert: Abnormal activity pattern detected in {tumorLocation || 'brain region'}
            </Typography>
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default BrainActivitySimulation;
