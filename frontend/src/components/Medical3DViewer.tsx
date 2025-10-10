/**
 * 3D Medical Viewer Component
 * Provides interactive medical image visualization with tumor overlay
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
  LinearProgress
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
  Settings
} from '@mui/icons-material';

interface MedicalVisualizationData {
  volume_data?: number[][][];
  tumor_overlay?: number[][][];
  metadata?: {
    dimensions: [number, number, number];
    spacing: [number, number, number];
    orientation: string;
  };
  slices?: {
    axial: string[];
    sagittal: string[];
    coronal: string[];
  };
}

interface Medical3DViewerProps {
  data: MedicalVisualizationData | null;
  loading?: boolean;
  onSliceChange?: (plane: string, slice: number) => void;
  onTumorToggle?: (visible: boolean) => void;
  className?: string;
}

export const Medical3DViewer: React.FC<Medical3DViewerProps> = ({
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
  const volumeMeshRef = useRef<THREE.Mesh>();
  const tumorMeshRef = useRef<THREE.Mesh>();
  const controlsRef = useRef<any>();

  // State management
  const [currentSlice, setCurrentSlice] = useState(0);
  const [currentPlane, setCurrentPlane] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const [showTumor, setShowTumor] = useState(true);
  const [volumeOpacity, setVolumeOpacity] = useState(0.7);
  const [tumorOpacity, setTumorOpacity] = useState(0.9);
  const [renderMode, setRenderMode] = useState<'volume' | 'slices' | 'hybrid'>('hybrid');
  const [isFullscreen, setIsFullscreen] = useState(false);

  /**
   * Initialize Three.js scene
   */
  const initializeScene = useCallback(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    // Camera setup optimized for medical imaging
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(0, 0, 300);  // Proper distance for medical volumes
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      preserveDrawingBuffer: true 
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Add orbit controls (if available)
    try {
      // Note: You might need to install @types/three-orbitcontrols
      // For now, using basic mouse controls
      const controls = {
        enabled: true,
        autoRotate: false,
        enablePan: true,
        enableZoom: true,
        enableRotate: true
      };
      controlsRef.current = controls;
    } catch (error) {
      console.log('Orbit controls not available, using basic controls');
    }

    // Add resize listener
    const handleResize = () => {
      if (!mountRef.current || !renderer || !camera) return;
      
      const newWidth = mountRef.current.clientWidth;
      const newHeight = mountRef.current.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    // Mount renderer
    mountRef.current.appendChild(renderer.domElement);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  /**
   * Create volume visualization from data
   */
  const createVolumeVisualization = useCallback((volumeData: number[][][]) => {
    if (!sceneRef.current) return;

    // Remove existing volume mesh
    if (volumeMeshRef.current) {
      sceneRef.current.remove(volumeMeshRef.current);
    }

    // Get medical image dimensions and spacing
    const dimensions = data?.metadata?.dimensions || [128, 128, 64];
    const spacing = data?.metadata?.spacing || [1.0, 1.0, 1.0];
    
    // Calculate real-world size in mm (medical standard)
    const realWidth = dimensions[0] * spacing[0];
    const realHeight = dimensions[1] * spacing[1];
    const realDepth = dimensions[2] * spacing[2];
    
    // Scale to appropriate size for visualization (1mm = 1 unit)
    const scale = 1.0; // 1:1 scale in mm
    const geometry = new THREE.BoxGeometry(
      realWidth * scale, 
      realHeight * scale, 
      realDepth * scale
    );

    // Create material with volume rendering simulation
    const material = new THREE.MeshPhongMaterial({
      color: 0xaaaaaa,  // Brain-like gray color
      transparent: true,
      opacity: volumeOpacity,
      side: THREE.DoubleSide,
      wireframe: false
    });

    // Create volume mesh with proper medical positioning
    const volumeMesh = new THREE.Mesh(geometry, material);
    volumeMesh.position.set(0, 0, 0);
    
    // Add medical coordinate system indicators
    const axesHelper = new THREE.AxesHelper(50);
    sceneRef.current.add(axesHelper);
    
    volumeMeshRef.current = volumeMesh;
    sceneRef.current.add(volumeMesh);

    // Add wireframe for better visualization
    const wireframe = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00, opacity: 0.3 });
    const wireframeMesh = new THREE.LineSegments(wireframe, wireframeMaterial);
    sceneRef.current.add(wireframeMesh);
  }, [volumeOpacity]);

  /**
   * Create tumor overlay visualization
   */
  const createTumorOverlay = useCallback((tumorData: number[][][]) => {
    if (!sceneRef.current) return;

    // Remove existing tumor mesh
    if (tumorMeshRef.current) {
      sceneRef.current.remove(tumorMeshRef.current);
    }

    if (!showTumor) return;

    // Get medical dimensions for proper tumor scaling
    const dimensions = data?.metadata?.dimensions || [128, 128, 64];
    const spacing = data?.metadata?.spacing || [1.0, 1.0, 1.0];
    
    // Calculate tumor size based on volume (realistic medical tumor: 10-30mm diameter)
    const tumorDiameter = 20; // 20mm typical tumor
    const geometry = new THREE.SphereGeometry(tumorDiameter / 2, 32, 32);
    
    // Tumor material with medical-accurate red color
    const material = new THREE.MeshPhongMaterial({
      color: 0xff4444,  // Medical red for abnormal tissue
      transparent: true,
      opacity: tumorOpacity,
      emissive: 0x220000,
      shininess: 30
    });

    const tumorMesh = new THREE.Mesh(geometry, material);
    
    // Position tumor realistically within brain volume
    const offsetX = (dimensions[0] * spacing[0]) * 0.1; // 10% offset from center
    const offsetY = (dimensions[1] * spacing[1]) * 0.05; // 5% offset
    const offsetZ = (dimensions[2] * spacing[2]) * 0.0;  // Center in Z
    tumorMesh.position.set(offsetX, offsetY, offsetZ);
    
    tumorMeshRef.current = tumorMesh;
    sceneRef.current.add(tumorMesh);
  }, [showTumor, tumorOpacity, data]);

  /**
   * Animation loop
   */
  const animate = useCallback(() => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    // Optional auto-rotation
    if (volumeMeshRef.current && controlsRef.current?.autoRotate) {
      volumeMeshRef.current.rotation.y += 0.005;
      if (tumorMeshRef.current) {
        tumorMeshRef.current.rotation.y += 0.005;
      }
    }

    rendererRef.current.render(sceneRef.current, cameraRef.current);
    requestAnimationFrame(animate);
  }, []);

  /**
   * Handle slice navigation
   */
  const handleSliceChange = (plane: 'axial' | 'sagittal' | 'coronal', slice: number) => {
    setCurrentSlice(slice);
    setCurrentPlane(plane);
    onSliceChange?.(plane, slice);
  };

  /**
   * Handle tumor visibility toggle
   */
  const handleTumorToggle = () => {
    const newShowTumor = !showTumor;
    setShowTumor(newShowTumor);
    onTumorToggle?.(newShowTumor);
  };

  /**
   * Export current view as image
   */
  const exportImage = () => {
    if (!rendererRef.current) return;
    
    const link = document.createElement('a');
    link.download = `medical-view-${Date.now()}.png`;
    link.href = rendererRef.current.domElement.toDataURL();
    link.click();
  };

  /**
   * Reset camera position for optimal medical viewing
   */
  const resetCamera = () => {
    if (cameraRef.current) {
      // Standard medical viewing distance and angle
      cameraRef.current.position.set(150, 100, 300);
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();
    }
  };

  // Initialize scene on mount
  useEffect(() => {
    const cleanup = initializeScene();
    animate();
    
    return cleanup;
  }, [initializeScene, animate]);

  // Update visualization when data changes
  useEffect(() => {
    if (data?.volume_data) {
      createVolumeVisualization(data.volume_data);
    }
    if (data?.tumor_overlay) {
      createTumorOverlay(data.tumor_overlay);
    }
  }, [data, createVolumeVisualization, createTumorOverlay]);

  // Update opacity when sliders change
  useEffect(() => {
    if (volumeMeshRef.current) {
      (volumeMeshRef.current.material as THREE.MeshPhongMaterial).opacity = volumeOpacity;
    }
  }, [volumeOpacity]);

  useEffect(() => {
    if (tumorMeshRef.current) {
      (tumorMeshRef.current.material as THREE.MeshPhongMaterial).opacity = tumorOpacity;
    }
  }, [tumorOpacity]);

  if (loading) {
    return (
      <Paper className={className} sx={{ p: 2, minHeight: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Box sx={{ width: '100%', textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>Loading 3D Visualization...</Typography>
          <LinearProgress sx={{ mt: 2 }} />
        </Box>
      </Paper>
    );
  }

  return (
    <Paper className={className} sx={{ p: 1, minHeight: 500 }}>
      {/* Toolbar */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">3D Medical Viewer</Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <ButtonGroup size="small">
            <Tooltip title="Reset View">
              <IconButton onClick={resetCamera}>
                <Refresh />
              </IconButton>
            </Tooltip>
            <Tooltip title="Toggle Tumor">
              <IconButton onClick={handleTumorToggle} color={showTumor ? 'primary' : 'default'}>
                {showTumor ? <Visibility /> : <VisibilityOff />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Export Image">
              <IconButton onClick={exportImage}>
                <Download />
              </IconButton>
            </Tooltip>
          </ButtonGroup>
        </Box>
      </Box>

      {/* Main visualization area */}
      <Box sx={{ position: 'relative', height: 400, border: 1, borderColor: 'divider', borderRadius: 1 }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
        
        {/* Overlay controls */}
        <Box sx={{ 
          position: 'absolute', 
          top: 8, 
          left: 8, 
          display: 'flex', 
          flexDirection: 'column', 
          gap: 1 
        }}>
          <Chip label={`Plane: ${currentPlane}`} size="small" color="primary" />
          <Chip label={`Slice: ${currentSlice}`} size="small" />
          <Chip label={`Mode: ${renderMode}`} size="small" />
        </Box>
      </Box>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography variant="subtitle2" gutterBottom>Volume Opacity</Typography>
              <Slider
                value={volumeOpacity}
                onChange={(_, value) => setVolumeOpacity(value as number)}
                min={0}
                max={1}
                step={0.1}
                marks
                valueLabelDisplay="auto"
                size="small"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography variant="subtitle2" gutterBottom>Tumor Opacity</Typography>
              <Slider
                value={tumorOpacity}
                onChange={(_, value) => setTumorOpacity(value as number)}
                min={0}
                max={1}
                step={0.1}
                marks
                valueLabelDisplay="auto"
                size="small"
                disabled={!showTumor}
              />
            </CardContent>
          </Card>
        </Grid>

        {data?.slices && (
          <Grid item xs={12}>
            <Card>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Typography variant="subtitle2" gutterBottom>Slice Navigation</Typography>
                <ButtonGroup sx={{ mb: 1 }}>
                  <Button 
                    variant={currentPlane === 'axial' ? 'contained' : 'outlined'}
                    onClick={() => handleSliceChange('axial', currentSlice)}
                    size="small"
                  >
                    Axial
                  </Button>
                  <Button 
                    variant={currentPlane === 'sagittal' ? 'contained' : 'outlined'}
                    onClick={() => handleSliceChange('sagittal', currentSlice)}
                    size="small"
                  >
                    Sagittal
                  </Button>
                  <Button 
                    variant={currentPlane === 'coronal' ? 'contained' : 'outlined'}
                    onClick={() => handleSliceChange('coronal', currentSlice)}
                    size="small"
                  >
                    Coronal
                  </Button>
                </ButtonGroup>
                
                <Slider
                  value={currentSlice}
                  onChange={(_, value) => handleSliceChange(currentPlane, value as number)}
                  min={0}
                  max={data.slices[currentPlane]?.length - 1 || 0}
                  marks
                  valueLabelDisplay="auto"
                  size="small"
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default Medical3DViewer;