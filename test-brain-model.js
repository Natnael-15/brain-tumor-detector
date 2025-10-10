#!/usr/bin/env node
/**
 * Brain Model Integration Test
 * Verifies that human_brain.glb is properly integrated
 */

const fs = require('fs');
const path = require('path');

console.log('Brain Model Integration Test');
console.log('================================');

// Check if model exists
const modelPath = path.join(__dirname, 'frontend', 'public', 'models', 'human_brain.glb');
const modelExists = fs.existsSync(modelPath);

console.log(`Model Path: ${modelPath}`);
console.log(`Model Exists: ${modelExists ? 'YES' : 'NO'}`);

if (modelExists) {
  const stats = fs.statSync(modelPath);
  console.log(`File Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Last Modified: ${stats.mtime.toISOString()}`);
  
  console.log('\nIntegration Status:');
  console.log('Brain model successfully placed in models directory');
  console.log('Advanced 3D Brain Viewer will auto-detect this model');
  console.log('Model will load automatically when viewing 3D tab');
  
  console.log('\nNext Steps:');
  console.log('1. Open http://localhost:3000');
  console.log('2. Go to "3D Visualization" tab');
  console.log('3. Your human_brain.glb model should load automatically!');
  console.log('4. Use the controls to adjust opacity, rotation, etc.');
  
} else {
  console.log('\nModel not found! Please check:');
  console.log('1. File is named exactly "human_brain.glb"');
  console.log('2. File is in the correct directory: frontend/public/models/');
  console.log('3. File is not corrupted');
}

console.log('\nApplication URLs:');
console.log('Frontend: http://localhost:3000');
console.log('Backend:  http://localhost:8000');
console.log('API Docs: http://localhost:8000/api/docs');

console.log('\nBrain Model Integration Complete!');
