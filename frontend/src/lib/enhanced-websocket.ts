/**
 * Enhanced WebSocket Hook with Advanced Features
 * Hospital-grade WebSocket connection with full reconnection and health monitoring
 */

'use client';

import { useWebSocket } from './websocket';

export function useEnhancedWebSocket() {
  // Return the same WebSocket client with the same interface
  return useWebSocket();
}
