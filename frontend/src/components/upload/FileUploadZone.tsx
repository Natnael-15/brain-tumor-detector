'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  InsertDriveFile as FileIcon,
} from '@mui/icons-material';

interface FileUploadZoneProps {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
  maxFiles?: number;
  maxSize?: number;
}

export function FileUploadZone({
  onFilesSelected,
  selectedFiles,
  maxFiles = 10,
  maxSize = 100 * 1024 * 1024, // 100MB
}: FileUploadZoneProps) {
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const validFiles = acceptedFiles.filter(file => file.size <= maxSize);
      const newFiles = [...selectedFiles, ...validFiles].slice(0, maxFiles);
      onFilesSelected(newFiles);
    },
    [selectedFiles, onFilesSelected, maxFiles, maxSize]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'application/octet-stream': ['.nii', '.nii.gz'],
      'image/*': ['.jpg', '.jpeg', '.png', '.tiff'],
    },
    maxFiles,
    maxSize,
  });

  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    onFilesSelected(newFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box>
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          borderRadius: 2,
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          bgcolor: isDragActive ? 'primary.50' : 'grey.50',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'primary.50',
          },
        }}
      >
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop files here' : 'Drag & drop medical images'}
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={2}>
          Or click to select files
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Supports DICOM (.dcm), NIfTI (.nii), JPEG, PNG, TIFF
        </Typography>
        <br />
        <Typography variant="caption" color="text.secondary">
          Max {maxFiles} files, {formatFileSize(maxSize)} each
        </Typography>
      </Box>

      {selectedFiles.length > 0 && (
        <Box mt={3}>
          <Typography variant="h6" gutterBottom>
            Selected Files ({selectedFiles.length})
          </Typography>
          <List>
            {selectedFiles.map((file, index) => (
              <ListItem key={index} divider>
                <FileIcon sx={{ mr: 2, color: 'primary.main' }} />
                <ListItemText
                  primary={file.name}
                  secondary={
                    <Box display="flex" gap={1} alignItems="center">
                      <Typography variant="caption" color="text.secondary">
                        {formatFileSize(file.size)}
                      </Typography>
                      <Chip
                        label={file.type || 'Unknown'}
                        size="small"
                        variant="outlined"
                        color="primary"
                      />
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    color="error"
                    onClick={() => removeFile(index)}
                    size="small"
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          
          {uploading && (
            <Box mt={2}>
              <Typography variant="body2" color="text.secondary" mb={1}>
                Processing files...
              </Typography>
              <LinearProgress />
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}