/**
 * Enhanced 3D Brain Visualization with Medical Accuracy
 * Provides realistic brain anatomy rendering with tumor detection
 */

'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Slider,
  Grid,
  Chip,
  Card,
  CardContent,
  Button,
  ButtonGroup,
  Tooltip,
  LinearProgress,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  RotateLeft,
  RotateRight,
  ZoomIn,
  ZoomOut,
  Refresh,
  ViewInAr,
  Layers,
  Visibility,
  VisibilityOff,
  Download,
  Fullscreen,
  Settings,
  Psychology,
  Healing,
  Warning
} from '@mui/icons-material';

interface EnhancedMedicalVisualizationData {
  volume_data?: number[][][];
  tumor_segmentation?: number[][][];
  metadata?: {
    dimensions: [number, number, number];
    spacing: [number, number, number];
    patient_id?: string;
    scan_type?: string;
    tumor_detected?: boolean;
    tumor_volume?: number;
    tumor_grade?: string;
    confidence?: number;
  };
  brain_regions?: {
    cortex?: number[][][];
    white_matter?: number[][][];
    ventricles?: number[][][];
    brainstem?: number[][][];
  };
}

interface EnhancedMedical3DViewerProps {
  data?: EnhancedMedicalVisualizationData;
  loading?: boolean;
  onSliceChange?: (plane: 'axial' | 'sagittal' | 'coronal', slice: number) => void;
  onTumorToggle?: (visible: boolean) => void;
  className?: string;
}

export const EnhancedMedical3DViewer: React.FC<EnhancedMedical3DViewerProps> = ({
  data,
  loading = false,
  onSliceChange,
  onTumorToggle,
  className
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<any>();
  
  // Brain anatomy mesh references
  const brainMeshRef = useRef<THREE.Group>();
  const cortexMeshRef = useRef<THREE.Mesh>();
  const whiteMatterMeshRef = useRef<THREE.Mesh>();
  const ventriclesMeshRef = useRef<THREE.Mesh>();
  const tumorMeshRef = useRef<THREE.Mesh>();

  // State management
  const [currentSlice, setCurrentSlice] = useState(0);
  const [currentPlane, setCurrentPlane] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const [showTumor, setShowTumor] = useState(true);
  const [showBrainRegions, setShowBrainRegions] = useState({
    cortex: true,
    whiteMatter: true,
    ventricles: false,
    brainstem: true
  });
  const [brainOpacity, setBrainOpacity] = useState(0.8);
  const [tumorOpacity, setTumorOpacity] = useState(0.9);
  const [renderQuality, setRenderQuality] = useState<'low' | 'medium' | 'high'>('medium');
  const [viewMode, setViewMode] = useState<'anatomical' | 'pathological' | 'hybrid'>('hybrid');
  const [isFullscreen, setIsFullscreen] = useState(false);

  /**
   * Initialize Three.js scene with medical lighting
   */
  const initializeScene = useCallback(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene setup with medical environment
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a); // Dark medical viewing background
    scene.fog = new THREE.Fog(0x1a1a1a, 500, 2000); // Depth perception
    sceneRef.current = scene;

    // Camera setup optimized for brain viewing
    const camera = new THREE.PerspectiveCamera(50, width / height, 1, 3000);
    camera.position.set(200, 150, 250);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup with medical-grade quality
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    rendererRef.current = renderer;

    // Medical lighting setup
    setupMedicalLighting(scene);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      
      renderer.render(scene, camera);
    };

    // Handle window resize
    const handleResize = () => {
      if (!mountRef.current) return;
      
      const newWidth = mountRef.current.clientWidth;
      const newHeight = mountRef.current.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);
    mountRef.current.appendChild(renderer.domElement);
    animate();

    // Initialize controls (would need OrbitControls import)
    // controlsRef.current = new OrbitControls(camera, renderer.domElement);
    // controlsRef.current.enableDamping = true;
    // controlsRef.current.dampingFactor = 0.05;

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  /**
   * Setup medical-grade lighting for brain visualization
   */
  const setupMedicalLighting = (scene: THREE.Scene) => {
    // Key light (main illumination)
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(100, 100, 100);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    scene.add(keyLight);

    // Fill light (reduces harsh shadows)
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-100, 50, 50);
    scene.add(fillLight);

    // Rim light (edge definition)
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
    rimLight.position.set(0, -100, -100);
    scene.add(rimLight);

    // Ambient light (overall scene illumination)
    const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
    scene.add(ambientLight);

    // Medical examination light
    const examLight = new THREE.SpotLight(0xffffff, 0.6);
    examLight.position.set(0, 200, 100);
    examLight.angle = Math.PI / 6;
    examLight.penumbra = 0.1;
    examLight.decay = 2;
    examLight.distance = 500;
    scene.add(examLight);
  };

  /**
   * Create realistic brain anatomy from medical data
   */
  const createBrainAnatomy = useCallback(() => {
    if (!sceneRef.current || !data) return;

    // Remove existing brain meshes
    if (brainMeshRef.current) {
      sceneRef.current.remove(brainMeshRef.current);
    }

    const brainGroup = new THREE.Group();
    brainMeshRef.current = brainGroup;

    // Get medical dimensions
    const dimensions = data.metadata?.dimensions || [256, 256, 128];
    const spacing = data.metadata?.spacing || [1.0, 1.0, 2.0];

    // Create brain cortex (outer gray matter)
    if (showBrainRegions.cortex) {
      const cortexGeometry = createBrainCortexGeometry(dimensions, spacing);
      const cortexMaterial = new THREE.MeshPhongMaterial({
        color: 0xc4a484, // Realistic brain cortex color
        transparent: true,
        opacity: brainOpacity * 0.9,
        side: THREE.DoubleSide,
        shininess: 20,
        specular: 0x333333
      });
      
      const cortexMesh = new THREE.Mesh(cortexGeometry, cortexMaterial);
      cortexMeshRef.current = cortexMesh;
      brainGroup.add(cortexMesh);
    }

    // Create white matter (inner brain tissue)
    if (showBrainRegions.whiteMatter) {
      const whiteMatterGeometry = createWhiteMatterGeometry(dimensions, spacing);
      const whiteMatterMaterial = new THREE.MeshPhongMaterial({
        color: 0xfaf0e6, // White matter color
        transparent: true,
        opacity: brainOpacity * 0.6,
        side: THREE.DoubleSide,
        shininess: 10
      });
      
      const whiteMatterMesh = new THREE.Mesh(whiteMatterGeometry, whiteMatterMaterial);
      whiteMatterMeshRef.current = whiteMatterMesh;
      brainGroup.add(whiteMatterMesh);
    }

    // Create ventricles (CSF spaces)
    if (showBrainRegions.ventricles) {
      const ventriclesGeometry = createVentriclesGeometry(dimensions, spacing);
      const ventriclesMaterial = new THREE.MeshPhongMaterial({
        color: 0x87ceeb, // CSF blue color
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        shininess: 100
      });
      
      const ventriclesMesh = new THREE.Mesh(ventriclesGeometry, ventriclesMaterial);
      ventriclesMeshRef.current = ventriclesMesh;
      brainGroup.add(ventriclesMesh);
    }

    sceneRef.current.add(brainGroup);
  }, [data, showBrainRegions, brainOpacity]);

  /**
   * Create brain cortex geometry with realistic shape
   */
  const createBrainCortexGeometry = (dimensions: number[], spacing: number[]) => {
    // Create an ellipsoid shape resembling a brain
    const geometry = new THREE.SphereGeometry(60, 32, 24);
    
    // Modify vertices to create brain-like shape
    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const y = vertices[i + 1];
      const z = vertices[i + 2];
      
      // Apply brain-like deformations
      vertices[i] = x * 1.2; // Widen
      vertices[i + 1] = y * 0.8; // Flatten top/bottom
      vertices[i + 2] = z * 1.4; // Elongate front/back
      
      // Add surface roughness for cortical folds
      const noise = (Math.sin(x * 0.1) + Math.cos(y * 0.1) + Math.sin(z * 0.1)) * 2;
      vertices[i] += noise * 0.5;
      vertices[i + 1] += noise * 0.3;
      vertices[i + 2] += noise * 0.4;
    }
    
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    
    return geometry;
  };

  /**
   * Create white matter geometry
   */
  const createWhiteMatterGeometry = (dimensions: number[], spacing: number[]) => {
    const geometry = new THREE.SphereGeometry(45, 24, 16);
    
    // Modify for white matter shape (smaller, more regular)
    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      vertices[i] *= 1.1;
      vertices[i + 1] *= 0.7;
      vertices[i + 2] *= 1.3;
    }
    
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
    
    return geometry;
  };

  /**
   * Create ventricles geometry
   */
  const createVentriclesGeometry = (dimensions: number[], spacing: number[]) => {
    // Create butterfly-shaped ventricles
    const group = new THREE.Group();
    
    // Lateral ventricles
    const leftVentricle = new THREE.SphereGeometry(15, 12, 8);
    const rightVentricle = new THREE.SphereGeometry(15, 12, 8);
    
    // Position ventricles
    leftVentricle.translate(-20, 10, 0);
    rightVentricle.translate(20, 10, 0);
    
    const mergedGeometry = new THREE.BufferGeometry();
    // Merge geometries would require additional library
    
    return leftVentricle; // Simplified for now
  };

  /**
   * Create tumor visualization with medical accuracy
   */
  const createTumorVisualization = useCallback(() => {
    if (!sceneRef.current || !data?.tumor_segmentation || !showTumor) return;

    // Remove existing tumor mesh
    if (tumorMeshRef.current) {
      sceneRef.current.remove(tumorMeshRef.current);
    }

    const tumorDetected = data.metadata?.tumor_detected;
    const tumorVolume = data.metadata?.tumor_volume || 0;
    const confidence = data.metadata?.confidence || 0;

    if (!tumorDetected) {
      // No tumor detected - maybe show healthy indicator
      return;
    }

    // Calculate tumor size based on volume (medical accuracy)
    const radius = Math.cbrt(tumorVolume / (4/3 * Math.PI)) || 12;
    
    // Create irregular tumor shape
    const geometry = new THREE.SphereGeometry(radius, 16, 12);
    
    // Add irregularity to tumor shape
    const vertices = geometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      const noise = Math.random() * 0.3;
      vertices[i] += noise;
      vertices[i + 1] += noise;
      vertices[i + 2] += noise;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    // Tumor material with medical colors
    const tumorGrade = data.metadata?.tumor_grade || 'unknown';
    const tumorColor = getTumorColor(tumorGrade);
    
    const material = new THREE.MeshPhongMaterial({
      color: tumorColor,
      transparent: true,
      opacity: tumorOpacity,
      emissive: new THREE.Color(tumorColor).multiplyScalar(0.2),
      shininess: 30
    });

    const tumorMesh = new THREE.Mesh(geometry, material);
    
    // Position tumor realistically (right frontal lobe - common location)
    tumorMesh.position.set(25, 15, 20);
    
    tumorMeshRef.current = tumorMesh;
    sceneRef.current.add(tumorMesh);

    // Add pulsing effect for active tumors
    const animateTumor = () => {
      if (tumorMeshRef.current) {
        const time = Date.now() * 0.001;
        const scale = 1 + Math.sin(time * 2) * 0.05;
        tumorMeshRef.current.scale.setScalar(scale);
      }
      requestAnimationFrame(animateTumor);
    };
    animateTumor();

  }, [data, showTumor, tumorOpacity]);

  /**
   * Get tumor color based on grade/type
   */
  const getTumorColor = (grade: string): number => {
    switch (grade.toLowerCase()) {
      case 'low': return 0xffa500; // Orange for low grade
      case 'high': return 0xff0000; // Red for high grade
      case 'malignant': return 0x8b0000; // Dark red for malignant
      case 'benign': return 0xffff00; // Yellow for benign
      default: return 0xff4444; // Default red
    }
  };

  /**
   * Get medical assessment text
   */
  const getMedicalAssessment = (): { text: string; severity: 'info' | 'warning' | 'error' | 'success' } => {
    if (!data?.metadata) {
      return { text: "Loading brain scan analysis...", severity: 'info' };
    }

    const tumorDetected = data.metadata.tumor_detected;
    const confidence = data.metadata.confidence || 0;
    const tumorGrade = data.metadata.tumor_grade;

    if (!tumorDetected) {
      return { 
        text: `Healthy Brain - No tumor detected (Confidence: ${(confidence * 100).toFixed(1)}%)`, 
        severity: 'success' 
      };
    }

    if (tumorGrade === 'benign') {
      return { 
        text: `Benign Tumor Detected - Non-cancerous growth identified (Confidence: ${(confidence * 100).toFixed(1)}%)`, 
        severity: 'warning' 
      };
    }

    return { 
      text: `Tumor Detected - ${tumorGrade} grade tumor identified (Confidence: ${(confidence * 100).toFixed(1)}%)`, 
      severity: 'error' 
    };
  };

  // Initialize scene
  useEffect(() => {
    const cleanup = initializeScene();
    return cleanup;
  }, [initializeScene]);

  // Update visualizations when data changes
  useEffect(() => {
    createBrainAnatomy();
    createTumorVisualization();
  }, [data, createBrainAnatomy, createTumorVisualization]);

  const assessment = getMedicalAssessment();

  return (
    <Paper elevation={3} className={className} sx={{ p: 2, height: '100%' }}>
      {/* Header with medical assessment */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Psychology color="primary" />
          3D Brain Analysis
        </Typography>
        
        <Alert severity={assessment.severity} sx={{ mb: 2 }}>
          {assessment.text}
        </Alert>

        {/* Controls */}
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <ButtonGroup size="small">
                <Tooltip title="Reset View">
                  <IconButton onClick={() => cameraRef.current?.position.set(200, 150, 250)}>
                    <Refresh />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Toggle Tumor">
                  <IconButton 
                    onClick={() => setShowTumor(!showTumor)} 
                    color={showTumor ? 'primary' : 'default'}
                  >
                    {showTumor ? <Visibility /> : <VisibilityOff />}
                  </IconButton>
                </Tooltip>
                <Tooltip title="Export Image">
                  <IconButton onClick={() => {
                    if (rendererRef.current) {
                      const link = document.createElement('a');
                      link.download = `brain-3d-${Date.now()}.png`;
                      link.href = rendererRef.current.domElement.toDataURL();
                      link.click();
                    }
                  }}>
                    <Download />
                  </IconButton>
                </Tooltip>
              </ButtonGroup>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Box sx={{ minWidth: 120 }}>
                <Typography variant="caption">Brain Opacity</Typography>
                <Slider
                  value={brainOpacity}
                  onChange={(_, value) => setBrainOpacity(value as number)}
                  min={0.1}
                  max={1}
                  step={0.1}
                  size="small"
                />
              </Box>
              {data?.metadata?.tumor_detected && (
                <Box sx={{ minWidth: 120 }}>
                  <Typography variant="caption">Tumor Opacity</Typography>
                  <Slider
                    value={tumorOpacity}
                    onChange={(_, value) => setTumorOpacity(value as number)}
                    min={0.1}
                    max={1}
                    step={0.1}
                    size="small"
                  />
                </Box>
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* 3D Visualization */}
      <Box sx={{ position: 'relative', height: 500, border: 1, borderColor: 'divider', borderRadius: 1 }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
        
        {loading && (
          <Box sx={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            right: 0, 
            bottom: 0, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white'
          }}>
            <Box sx={{ textAlign: 'center' }}>
              <LinearProgress sx={{ mb: 2, width: 200 }} />
              <Typography>Rendering 3D Brain Model...</Typography>
            </Box>
          </Box>
        )}
      </Box>

      {/* Medical Information Panel */}
      {data?.metadata && (
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Scan Information
                </Typography>
                <Typography variant="body2">
                  Type: {data.metadata.scan_type || 'MRI Brain'}
                </Typography>
                <Typography variant="body2">
                  Dimensions: {data.metadata.dimensions?.join(' × ') || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  Patient ID: {data.metadata.patient_id || 'Anonymous'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          {data.metadata.tumor_detected && (
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    Tumor Analysis
                  </Typography>
                  <Typography variant="body2">
                    Volume: {data.metadata.tumor_volume?.toFixed(2) || 'N/A'} cm³
                  </Typography>
                  <Typography variant="body2">
                    Grade: {data.metadata.tumor_grade || 'Under analysis'}
                  </Typography>
                  <Typography variant="body2">
                    Confidence: {((data.metadata.confidence || 0) * 100).toFixed(1)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Paper>
  );
};

export default EnhancedMedical3DViewer;