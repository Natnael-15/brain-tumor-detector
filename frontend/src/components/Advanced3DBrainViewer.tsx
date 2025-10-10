/**
 * Advanced 3D Brain Viewer - Simple Spinning Ball (Hydration Fix)
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
  Chip,
  LinearProgress
} from '@mui/material';
import {
  ViewInAr,
  Settings,
  Download,
  Refresh,
  Visibility,
  VisibilityOff
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
  const brainModelRef = useRef<THREE.Group>();
  const tumorGroupRef = useRef<THREE.Group>();
  const mixerRef = useRef<THREE.AnimationMixer>();

  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [brainOpacity, setBrainOpacity] = useState(0.8);
  const [showTumors, setShowTumors] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  /**
   * Initialize Three.js Scene
   */
  const initScene = () => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
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
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 2;
    controls.maxDistance = 20;
    controls.maxPolarAngle = Math.PI;
    controlsRef.current = controls;

    // Lighting setup for medical visualization
    setupMedicalLighting(scene);

    // Groups for organization
    brainModelRef.current = new THREE.Group();
    brainModelRef.current.name = 'BrainModel';
    scene.add(brainModelRef.current);

    tumorGroupRef.current = new THREE.Group();
    tumorGroupRef.current.name = 'TumorGroup';
    scene.add(tumorGroupRef.current);

    mountRef.current.appendChild(renderer.domElement);

    // Check for available models
    checkAvailableModels();
  };

  /**
   * Setup medical-grade lighting
   */
  const setupMedicalLighting = (scene: THREE.Scene) => {
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    // Main directional light
    const mainLight = new THREE.DirectionalLight(0xffffff, 1);
    mainLight.position.set(5, 5, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    scene.add(mainLight);

    // Fill lights for better medical visualization
    const fillLight1 = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight1.position.set(-5, 3, 2);
    scene.add(fillLight1);

    const fillLight2 = new THREE.DirectionalLight(0xffffff, 0.2);
    fillLight2.position.set(2, -3, -5);
    scene.add(fillLight2);

    // Rim light for depth
    const rimLight = new THREE.DirectionalLight(0x6699ff, 0.4);
    rimLight.position.set(-5, 5, -5);
    scene.add(rimLight);
  };

  /**
   * Check what brain models are available in the public folder
   */
  const checkAvailableModels = async () => {
    try {
      // First check if there's a models configuration file
      const configResponse = await fetch('/models/models.json');
      if (configResponse.ok) {
        const config = await configResponse.json();
        if (config.available_models && config.available_models.length > 0) {
          console.log(`🧠 Found ${config.available_models.length} brain models from config`);
          setAvailableModels(config.available_models);
          return;
        }
      }
    } catch (error) {
      // Config file not found, proceed with reduced model checking
    }

    // Disable model checking to prevent 404 spam
    // Use procedural brain geometry instead
    console.log('🧠 No brain models configured, using procedural geometry');
    setAvailableModels([]);
    
    // Fallback to procedural brain since no model checking
    console.log('⚠️ No external brain models found, creating procedural brain');
    createProceduralBrain();
  };

  /**
   * Load external brain model (GLB, GLTF, OBJ, FBX)
   */
  const loadBrainModel = async (modelPath: string) => {
    if (!sceneRef.current || !brainModelRef.current) return;

    setIsLoading(true);
    setError(null);
    setLoadingProgress(0);

    try {
      const fileExtension = modelPath.split('.').pop()?.toLowerCase();
      let loader: GLTFLoader | OBJLoader | FBXLoader;
      
      // Progress tracking
      const onProgress = (progress: ProgressEvent) => {
        if (progress.lengthComputable) {
          const percentComplete = (progress.loaded / progress.total) * 100;
          setLoadingProgress(percentComplete);
        }
      };

      switch (fileExtension) {
        case 'glb':
        case 'gltf':
          loader = new GLTFLoader();
          break;
        case 'obj':
          // Check for corresponding MTL file
          const mtlPath = modelPath.replace('.obj', '.mtl');
          try {
            const mtlLoader = new MTLLoader();
            const materials = await new Promise<any>((resolve, reject) => {
              mtlLoader.load(mtlPath, resolve, onProgress, reject);
            });
            materials.preload();
            loader = new OBJLoader();
            (loader as OBJLoader).setMaterials(materials);
          } catch {
            loader = new OBJLoader();
          }
          break;
        case 'fbx':
          loader = new FBXLoader();
          break;
        default:
          throw new Error(`Unsupported file format: ${fileExtension}`);
      }

      // Load the model
      const loadedModel = await new Promise<any>((resolve, reject) => {
        loader.load(modelPath, resolve, onProgress, reject);
      });

      // Clear existing model
      brainModelRef.current.clear();

      // Process loaded model
      let brainMesh: THREE.Object3D;
      
      if (fileExtension === 'glb' || fileExtension === 'gltf') {
        brainMesh = loadedModel.scene;
        
        // Setup animations if available
        if (loadedModel.animations && loadedModel.animations.length > 0) {
          mixerRef.current = new THREE.AnimationMixer(brainMesh);
          loadedModel.animations.forEach((clip: THREE.AnimationClip) => {
            mixerRef.current?.clipAction(clip).play();
          });
        }
      } else {
        brainMesh = loadedModel;
      }

      // Apply medical brain material
      applyMedicalBrainMaterial(brainMesh);

      // Scale and position the model
      const box = new THREE.Box3().setFromObject(brainMesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2 / maxDim; // Scale to fit in viewport

      brainMesh.scale.setScalar(scale);
      brainMesh.position.sub(center.multiplyScalar(scale));

      brainModelRef.current.add(brainMesh);
      
      setModelLoaded(true);
      setIsLoading(false);
      setLoadingProgress(100);

      // Generate tumors based on analysis data
      if (data && data.hasTumor) {
        generateTumors();
      }

    } catch (error) {
      console.error('Error loading brain model:', error);
      setError(`Failed to load brain model: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsLoading(false);
      
      // Fallback to procedural brain
      createProceduralBrain();
    }
  };

  /**
   * Apply medical-grade material to brain model
   */
  const applyMedicalBrainMaterial = (object: THREE.Object3D) => {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        // Create medical brain material
        const material = new THREE.MeshPhongMaterial({
          color: 0xffb3d1, // Realistic brain pink
          transparent: true,
          opacity: brainOpacity,
          shininess: 10,
          specular: 0x444444,
          wireframe: wireframe
        });

        child.material = material;
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  };

  /**
   * Create procedural brain (fallback if no model available)
   */
  const createProceduralBrain = () => {
    if (!brainModelRef.current) return;

    console.log('🧠 Creating procedural brain as fallback');

    // Create more realistic brain geometry
    const brainGeometry = new THREE.SphereGeometry(1, 64, 32);
    
    // Apply noise for brain-like surface
    const positions = brainGeometry.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);
      
      // Add noise for realistic brain surface
      const noise = (Math.sin(x * 5) + Math.sin(y * 7) + Math.sin(z * 6)) * 0.1;
      const length = Math.sqrt(x * x + y * y + z * z);
      const factor = (1 + noise) / length;
      
      positions.setXYZ(i, x * factor, y * factor, z * factor);
    }
    positions.needsUpdate = true;
    brainGeometry.computeVertexNormals();

    const brainMaterial = new THREE.MeshPhongMaterial({
      color: 0xffb3d1,
      transparent: true,
      opacity: brainOpacity,
      shininess: 10,
      wireframe: wireframe
    });

    const brainMesh = new THREE.Mesh(brainGeometry, brainMaterial);
    brainMesh.castShadow = true;
    brainMesh.receiveShadow = true;

    brainModelRef.current.add(brainMesh);
    setModelLoaded(true);

    if (data && data.hasTumor) {
      generateTumors();
    }
  };

  /**
   * Generate tumor visualizations
   */
  const generateTumors = () => {
    if (!tumorGroupRef.current || !data) return;

    tumorGroupRef.current.clear();

    // Create tumor based on analysis data
    const tumorGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    const tumorMaterial = new THREE.MeshPhongMaterial({
      color: 0xff4444,
      transparent: true,
      opacity: 0.8,
      emissive: 0x441111,
      emissiveIntensity: 0.5
    });

    const tumor = new THREE.Mesh(tumorGeometry, tumorMaterial);
    tumor.position.set(
      (Math.random() - 0.5) * 1.5,
      (Math.random() - 0.5) * 1.5,
      (Math.random() - 0.5) * 1.5
    );

    // Add pulsing animation
    const tumorScale = { scale: 1 };
    const pulseTween = () => {
      const duration = 2000;
      const startTime = Date.now();
      
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = (elapsed % duration) / duration;
        const scale = 1 + Math.sin(progress * Math.PI * 2) * 0.1;
        
        tumor.scale.setScalar(scale);
        
        if (modelLoaded) {
          requestAnimationFrame(animate);
        }
      };
      animate();
    };
    pulseTween();

    // Add click handler
    tumor.userData = { 
      type: 'tumor', 
      data: data,
      onClick: () => onTumorClick?.(data)
    };

    tumorGroupRef.current.add(tumor);
  };

  /**
   * Animation loop
   */
  const animate = () => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    // Update controls
    controlsRef.current?.update();

    // Update animations
    if (mixerRef.current) {
      mixerRef.current.update(0.016); // ~60fps
    }

    // Auto rotate
    if (autoRotate && brainModelRef.current) {
      brainModelRef.current.rotation.y += 0.005;
    }

    // Render
    rendererRef.current.render(sceneRef.current, cameraRef.current);
    
    if (modelLoaded) {
      requestAnimationFrame(animate);
    }
  };

  /**
   * Handle window resize
   */
  const handleResize = () => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  };

  // Effects
  useEffect(() => {
    initScene();
    animate();

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (rendererRef.current && mountRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  // Update brain opacity
  useEffect(() => {
    if (brainModelRef.current) {
      brainModelRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh && child.material instanceof THREE.Material) {
          (child.material as any).opacity = brainOpacity;
          (child.material as any).wireframe = wireframe;
        }
      });
    }
  }, [brainOpacity, wireframe]);

  // Toggle tumor visibility
  useEffect(() => {
    if (tumorGroupRef.current) {
      tumorGroupRef.current.visible = showTumors;
    }
  }, [showTumors]);

  // Auto rotation
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
    }
  }, [autoRotate]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Loading Overlay */}
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.7)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10,
            color: 'white'
          }}
        >
          <Typography variant="h6" gutterBottom>
            Loading 3D Brain Model...
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={loadingProgress} 
            sx={{ width: '300px', mb: 2 }}
          />
          <Typography variant="body2">
            {Math.round(loadingProgress)}%
          </Typography>
        </Box>
      )}

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="warning" 
          sx={{ position: 'absolute', top: 10, left: 10, right: 10, zIndex: 5 }}
          action={
            <Button color="inherit" size="small" onClick={() => setError(null)}>
              DISMISS
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {/* Controls Panel */}
      <Paper
        sx={{
          position: 'absolute',
          top: 10,
          right: 10,
          p: 2,
          minWidth: 200,
          zIndex: 5,
          backgroundColor: 'rgba(0,0,0,0.8)',
          color: 'white'
        }}
      >
        <Typography variant="subtitle1" gutterBottom>
          <ViewInAr sx={{ mr: 1, verticalAlign: 'middle' }} />
          3D Brain Controls
        </Typography>

        {/* Model Status */}
        <Box sx={{ mb: 2 }}>
          <Chip
            icon={modelLoaded ? <Visibility /> : <VisibilityOff />}
            label={modelLoaded ? "Model Loaded" : "No Model"}
            color={modelLoaded ? "success" : "default"}
            size="small"
          />
          {availableModels.length > 0 && (
            <Chip
              label={`${availableModels.length} Models Found`}
              color="info"
              size="small"
              sx={{ ml: 1 }}
            />
          )}
        </Box>

        {/* Brain Opacity */}
        <Typography variant="body2" gutterBottom>
          Brain Opacity
        </Typography>
        <Slider
          value={brainOpacity}
          onChange={(_, value) => setBrainOpacity(value as number)}
          min={0.1}
          max={1}
          step={0.1}
          size="small"
          sx={{ mb: 2 }}
        />

        {/* Toggle Controls */}
        <FormControlLabel
          control={
            <Switch
              checked={showTumors}
              onChange={(e) => setShowTumors(e.target.checked)}
              size="small"
            />
          }
          label="Show Tumors"
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
        />

        <FormControlLabel
          control={
            <Switch
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              size="small"
            />
          }
          label="Auto Rotate"
        />

        {/* Model Selection */}
        {availableModels.length > 1 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Available Models:
            </Typography>
            {availableModels.map((model, index) => (
              <Button
                key={index}
                size="small"
                variant="outlined"
                onClick={() => loadBrainModel(model)}
                sx={{ mb: 1, mr: 1, color: 'white', borderColor: 'white' }}
              >
                {model.split('/').pop()}
              </Button>
            ))}
          </Box>
        )}

        {/* Refresh Button */}
        <Button
          startIcon={<Refresh />}
          onClick={checkAvailableModels}
          variant="outlined"
          size="small"
          fullWidth
          sx={{ mt: 1, color: 'white', borderColor: 'white' }}
        >
          Refresh Models
        </Button>
      </Paper>

      {/* 3D Viewport */}
      <div 
        ref={mountRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          minHeight: '400px',
          cursor: 'grab'
        }} 
      />

      {/* Instructions */}
      {modelLoaded && (
        <Paper
          sx={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            p: 2,
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white'
          }}
        >
          <Typography variant="body2">
            🖱️ Left click + drag: Rotate • 🔍 Scroll: Zoom • 🖱️ Right click + drag: Pan
          </Typography>
        </Paper>
      )}
    </Box>
  );
}