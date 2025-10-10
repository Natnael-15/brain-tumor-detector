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

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000011);
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

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0x00ff88, 0.5, 100);
    pointLight.position.set(-10, -10, -10);
    scene.add(pointLight);

    // Create brain sphere (simplified)
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: 0xffc0cb,
      shininess: 30,
      wireframe: wireframe
    });
    
    const brain = new THREE.Mesh(geometry, material);
    brain.castShadow = true;
    brain.receiveShadow = true;
    scene.add(brain);
    brainRef.current = brain;

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
      (brainRef.current.material as THREE.MeshPhongMaterial).wireframe = wireframe;
    }
  }, [wireframe]);

  const addTumorMarkers = (scene: THREE.Scene, analysisData: any) => {
    // Add red spheres to represent tumors
    const tumorGeometry = new THREE.SphereGeometry(0.2, 16, 16);
    const tumorMaterial = new THREE.MeshPhongMaterial({
      color: 0xff0000,
      emissive: 0x330000
    });

    // Add random tumor positions for demonstration
    const tumorPositions = [
      { x: 1, y: 0.5, z: 0 },
      { x: -0.8, y: -0.3, z: 0.7 },
      { x: 0.2, y: 1.2, z: -0.5 }
    ];

    tumorPositions.forEach((pos, index) => {
      const tumor = new THREE.Mesh(tumorGeometry, tumorMaterial);
      tumor.position.set(pos.x, pos.y, pos.z);
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

      {/* Controls Panel */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          background: 'rgba(255,255,255,0.9)',
          backdropFilter: 'blur(10px)',
          borderRadius: 2,
          p: 2,
          minWidth: 200
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <ViewInAr color="primary" />
          <Typography variant="subtitle2" fontWeight="bold">
            3D Brain Viewer
          </Typography>
        </Box>

        <FormControlLabel
          control={
            <Switch
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              size="small"
            />
          }
          label="Auto Rotate"
          sx={{ mb: 1 }}
        />

        <FormControlLabel
          control={
            <Switch
              checked={wireframe}
              onChange={(e) => setWireframe(e.target.checked)}
              size="small"
            />
          }
          label="Wireframe"
          sx={{ mb: 2 }}
        />

        <Typography variant="caption" gutterBottom>
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
        />

        <Button
          fullWidth
          variant="outlined"
          size="small"
          startIcon={<Refresh />}
          onClick={resetView}
        >
          Reset View
        </Button>
      </Box>

      {/* Info Panel */}
      {data && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(255,255,255,0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: 2,
            p: 2
          }}
        >
          <Typography variant="subtitle2" gutterBottom>
            Analysis Results
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label={data.tumor_detected ? 'Tumor Detected' : 'No Tumor'}
              color={data.tumor_detected ? 'error' : 'success'}
              size="small"
            />
            {data.confidence && (
              <Chip
                label={`${(data.confidence * 100).toFixed(1)}% Confidence`}
                variant="outlined"
                size="small"
              />
            )}
          </Box>
        </Box>
      )}
    </Paper>
  );
}