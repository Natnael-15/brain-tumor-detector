'use client';

import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0969da',
      light: '#54a3ff',
      dark: '#0550ae',
    },
    secondary: {
      main: '#24292f',
      light: '#57606a',
      dark: '#0e1115',
    },
    background: {
      default: '#f6f8fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  }
});
export default theme;
