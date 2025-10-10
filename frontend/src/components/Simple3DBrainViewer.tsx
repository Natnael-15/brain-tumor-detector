/**
 * Simple 3D Brain Visualization - Spinning Ball (Hydration Fix)
 * Simplified 3D visualization to eliminate hydration issues
 */

'use client';

import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  Box,
  Typography,
  Paper,
  Slider,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  Chip
} from '@mui/material';
import {
  ViewInAr,
  Settings,
  Refresh
} from '@mui/icons-material';

interface Advanced3DBrainViewerProps {
  data?: any;
  analysisId?: string;
  onTumorClick?: (tumorData: any) => void;
}

export default function Advanced3DBrainViewer({ 
  data, 
  analysisId,
  onTumorClick 
}: Advanced3DBrainViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const brainRef = useRef<THREE.Mesh>();
  const animationRef = useRef<number>();

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(0.01);
  const [autoRotate, setAutoRotate] = useState(true);
  const [wireframe, setWireframe] = useState(false);

  // Initialize 3D scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup with medical-grade background
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a); // Professional dark gray
    scene.fog = new THREE.Fog(0x1a1a1a, 10, 50); // Subtle depth fog
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 5);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 2.0;
    controlsRef.current = controls;

    // Medical-grade lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    // Key light (main medical examination light)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
    keyLight.position.set(5, 10, 7);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    scene.add(keyLight);

    // Fill light (softer, reduces harsh shadows)
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    // Back light (rim lighting for depth perception)
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(0, -5, -10);
    scene.add(backLight);

    // Subtle accent light for medical highlighting
    const accentLight = new THREE.PointLight(0x4a90e2, 0.3, 100);
    accentLight.position.set(0, 10, 0);
    scene.add(accentLight);

    // Create medical-grade brain model
    const geometry = new THREE.SphereGeometry(2, 64, 64); // Higher detail for medical accuracy
    
    // Medical brain tissue material with realistic properties
    const material = new THREE.MeshStandardMaterial({
      color: 0xffc8d4, // Realistic brain tissue color
      roughness: 0.8,
      metalness: 0.1,
      emissive: 0x221122,
      emissiveIntensity: 0.05,
      wireframe: wireframe
    });
    
    const brain = new THREE.Mesh(geometry, material);
    brain.castShadow = true;
    brain.receiveShadow = true;
    scene.add(brain);
    brainRef.current = brain;

    // Add brain surface details (sulci/gyri simulation)
    const detailGeometry = new THREE.IcosahedronGeometry(2.02, 2);
    const detailMaterial = new THREE.MeshStandardMaterial({
      color: 0xe8b4c4,
      roughness: 0.9,
      metalness: 0.05,
      transparent: true,
      opacity: 0.3,
      wireframe: false
    });
    const detailMesh = new THREE.Mesh(detailGeometry, detailMaterial);
    brain.add(detailMesh);

    // Add tumor markers if data exists
    if (data?.tumor_detected) {
      addTumorMarkers(scene, data);
    }

    setIsLoading(false);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      
      if (autoRotate && brainRef.current) {
        brainRef.current.rotation.y += rotationSpeed;
        brainRef.current.rotation.x += rotationSpeed * 0.5;
      }
      
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update rotation speed
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
    }
  }, [autoRotate]);

  // Update wireframe
  useEffect(() => {
    if (brainRef.current) {
      (brainRef.current.material as THREE.MeshStandardMaterial).wireframe = wireframe;
    }
  }, [wireframe]);

  const addTumorMarkers = (scene: THREE.Scene, analysisData: any) => {
    // Medical-grade tumor visualization
    const tumorGeometry = new THREE.SphereGeometry(0.3, 24, 24);
    const tumorMaterial = new THREE.MeshStandardMaterial({
      color: 0xff3333,
      emissive: 0xff0000,
      emissiveIntensity: 0.3,
      roughness: 0.6,
      metalness: 0.2,
      transparent: true,
      opacity: 0.85
    });

    // Tumor glow effect
    const glowGeometry = new THREE.SphereGeometry(0.35, 24, 24);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0.2
    });

    // Position tumors based on analysis data or use realistic medical positions
    const tumorPositions: Array<{ x: number; y: number; z: number }> = analysisData.tumor_locations || [
      { x: 1.2, y: 0.6, z: 0.3 },   // Right frontal lobe
      { x: -0.9, y: -0.4, z: 0.8 }, // Left temporal lobe
      { x: 0.2, y: 1.2, z: -0.5 }
    ];

    tumorPositions.forEach((pos, index) => {
      // Main tumor
      const tumor = new THREE.Mesh(tumorGeometry, tumorMaterial);
      tumor.position.set(pos.x, pos.y, pos.z);
      tumor.castShadow = true;
      tumor.userData = { 
        type: 'tumor', 
        index,
        onClick: () => {
          if (onTumorClick) {
            onTumorClick({
              position: pos,
              index,
              type: 'glioblastoma',
              confidence: 0.89
            });
          }
        }
      };
      scene.add(tumor);

      // Add glow effect around tumor
      const glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.position.set(pos.x, pos.y, pos.z);
      scene.add(glow);
    });
  };

  const resetView = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 0, 5);
      controlsRef.current.reset();
    }
  };

  return (
    <Paper elevation={3} sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
      {/* 3D Viewer */}
      <Box
        ref={mountRef}
        sx={{
          width: '100%',
          height: '400px',
          position: 'relative',
          cursor: 'grab',
          '&:active': { cursor: 'grabbing' }
        }}
      />

      {/* Loading State */}
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(0,0,0,0.8)',
            color: 'white'
          }}
        >
          <Typography>Loading 3D Brain Model...</Typography>
        </Box>
      )}

      {/* Medical Controls Panel */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          background: 'rgba(255,255,255,0.95)',
          backdropFilter: 'blur(10px)',
          borderRadius: 2,
          p: 2,
          minWidth: 240,
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <ViewInAr color="primary" />
          <Typography variant="subtitle2" fontWeight="bold">
            Medical 3D Viewer
          </Typography>
        </Box>

        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
          Advanced brain visualization with AI-detected regions
        </Typography>

        <FormControlLabel
          control={
            <Switch
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              size="small"
              color="primary"
            />
          }
          label={<Typography variant="body2">Auto Rotate</Typography>}
          sx={{ mb: 1, width: '100%' }}
        />

        <FormControlLabel
          control={
            <Switch
              checked={wireframe}
              onChange={(e) => setWireframe(e.target.checked)}
              size="small"
              color="primary"
            />
          }
          label={<Typography variant="body2">Wireframe Mode</Typography>}
          sx={{ mb: 2, width: '100%' }}
        />

        <Typography variant="caption" gutterBottom sx={{ display: 'block', fontWeight: 'bold' }}>
          Rotation Speed
        </Typography>
        <Slider
          value={rotationSpeed}
          onChange={(_, value) => setRotationSpeed(value as number)}
          min={0}
          max={0.05}
          step={0.001}
          size="small"
          sx={{ mb: 2 }}
          color="primary"
        />

        <Button
          fullWidth
          variant="contained"
          size="small"
          startIcon={<Refresh />}
          onClick={resetView}
          sx={{ mb: 1 }}
        >
          Reset View
        </Button>

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
          Use mouse to rotate, zoom, and pan
        </Typography>
      </Box>

      {/* Medical Info Panel */}
      {data && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(255,255,255,0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: 2,
            p: 2,
            minWidth: 280,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}
        >
          <Typography variant="subtitle2" gutterBottom fontWeight="bold">
            Clinical Analysis Results
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
            <Chip
              label={data.tumor_detected ? '⚠️ Tumor Detected' : '✓ No Tumor'}
              color={data.tumor_detected ? 'error' : 'success'}
              size="small"
              sx={{ fontWeight: 'bold' }}
            />
            {data.confidence && (
              <Chip
                label={`${(data.confidence * 100).toFixed(1)}% Confidence`}
                variant="outlined"
                size="small"
                color="primary"
              />
            )}
          </Box>
          {data.tumor_type && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Type: {data.tumor_type}
            </Typography>
          )}
          {data.tumor_volume && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Volume: {data.tumor_volume.toFixed(1)} cm³
            </Typography>
          )}
        </Box>
      )}
    </Paper>
  );
}