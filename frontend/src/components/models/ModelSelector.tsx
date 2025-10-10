'use client';

import React from 'react';
import {
  Box,
  Typography,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Chip,
  Card,
  CardContent,
} from '@mui/material';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

interface ModelOption {
  id: string;
  name: string;
  description: string;
  type: string;
  status: 'ready' | 'loading' | 'error';
}

const models: ModelOption[] = [
  {
    id: 'ensemble',
    name: 'Advanced Ensemble Model',
    description: 'Multi-model ensemble with uncertainty quantification and attention mechanisms',
    type: 'ensemble',
    status: 'ready',
  },
  {
    id: 'advanced_unet',
    name: 'Advanced 3D U-Net',
    description: 'Enhanced U-Net with spatial/channel attention and deep supervision',
    type: 'segmentation',
    status: 'ready',
  },
  {
    id: 'medical_vit',
    name: 'Medical Vision Transformer',
    description: '3D ViT optimized for medical imaging with spatial awareness',
    type: 'classification',
    status: 'ready',
  },
  {
    id: 'nnunet',
    name: 'nnU-Net',
    description: 'State-of-the-art medical segmentation with automated preprocessing',
    type: 'segmentation',
    status: 'ready',
  },
  {
    id: 'unet3d',
    name: 'Legacy U-Net 3D',
    description: 'Classic 3D convolutional neural network',
    type: 'segmentation',
    status: 'ready',
  },
];

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Select AI Model
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={2}>
        Choose the AI model for tumor detection and analysis
      </Typography>
      
      <FormControl component="fieldset" fullWidth>
        <RadioGroup
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {models.map((model) => (
            <Card
              key={model.id}
              variant="outlined"
              sx={{
                mb: 1,
                cursor: 'pointer',
                bgcolor: selectedModel === model.id ? 'primary.50' : 'transparent',
                borderColor: selectedModel === model.id ? 'primary.main' : 'grey.300',
                '&:hover': {
                  bgcolor: 'primary.50',
                  borderColor: 'primary.main',
                },
              }}
              onClick={() => onModelChange(model.id)}
            >
              <CardContent sx={{ py: 2 }}>
                <FormControlLabel
                  value={model.id}
                  control={<Radio color="primary" />}
                  label={
                    <Box>
                      <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                        <Typography variant="subtitle1" fontWeight={500}>
                          {model.name}
                        </Typography>
                        <Chip
                          label={model.type}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                        <Chip
                          label={model.status}
                          size="small"
                          color={model.status === 'ready' ? 'success' : 'warning'}
                          variant="filled"
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {model.description}
                      </Typography>
                    </Box>
                  }
                  sx={{ m: 0, width: '100%' }}
                />
              </CardContent>
            </Card>
          ))}
        </RadioGroup>
      </FormControl>
    </Box>
  );
}