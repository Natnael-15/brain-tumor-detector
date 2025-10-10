import { TableBody, TableBodyProps } from '@mui/material';
import React from 'react';

// Debug wrapper to help identify hydration issues
export const DebugTableBody: React.FC<TableBodyProps> = (props) => {
  return <TableBody suppressHydrationWarning {...props} />;
};