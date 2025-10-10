'use client';

import { createTheme } from '@mui/material/styles';

// Create Material-UI theme for medical application
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2', // Medical blue
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e', // Medical red for alerts
      light: '#f06292',
      dark: '#ad1457',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
    background: {
      default: '#fafafa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    h1: {
      fontWeight: 600,
      fontSize: '2.5rem',
      color: '#1976d2',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      color: '#1976d2',
    },
    h3: {
      fontWeight: 500,
      fontSize: '1.75rem',
      color: '#1976d2',
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.5rem',
      color: '#333',
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.25rem',
      color: '#333',
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
      color: '#333',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
        contained: {
          boxShadow: '0 2px 8px rgba(25, 118, 210, 0.3)',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(25, 118, 210, 0.4)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          '&:hover': {
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '1rem',
        },
      },
    },
  },
});

export default theme;