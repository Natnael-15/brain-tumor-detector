/**
 * WebSocket Hook for Real-time Communication
 * Medical-grade WebSocket connection with auto-reconnection
 */

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

export interface AnalysisUpdate {
  type: string;
  analysis_id: string;
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
  model?: string;
  file_name?: string;
  results?: any;
  status?: string;
}

type MessageCallback = (data: AnalysisUpdate) => void;
type ConnectionCallback = (connected: boolean) => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private messageCallbacks: Set<MessageCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();
  private userId: string | null = null;
  private isManualClose = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  constructor() {
    this.url = 'ws://localhost:8000';
  }

  /**
   * Connect to WebSocket server
   */
  async connect(userId?: string): Promise<boolean> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log(' WebSocket already connected');
      return true;
    }

    this.isManualClose = false;
    this.userId = userId || this.userId || `user_${Date.now()}`;
    
    return new Promise((resolve) => {
      try {
        const wsUrl = `${this.url}/ws/${this.userId}`;
        console.log('🔗 Connecting to WebSocket:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log(' WebSocket connected successfully');
          this.reconnectAttempts = 0;
          this.notifyConnectionChange(true);
          this.startHeartbeat();
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📨 WebSocket message received:', data);
            
            // Handle different message types
            if (data.type === 'connection_established') {
              console.log(' Connection established:', data);
            } else if (data.type === 'analysis_update') {
              this.notifyMessageReceived(data);
            } else if (data.type === 'pong') {
              console.log('💓 Heartbeat pong received');
            } else {
              // Forward all messages to callbacks
              this.notifyMessageReceived(data);
            }
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          resolve(false);
        };

        this.ws.onclose = () => {
          console.log('🔴 WebSocket closed');
          this.stopHeartbeat();
          this.notifyConnectionChange(false);
          
          if (!this.isManualClose && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        // Timeout if connection takes too long
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.warn('⚠️ WebSocket connection timeout');
            resolve(false);
          }
        }, 5000);

      } catch (error) {
        console.error('❌ Error creating WebSocket:', error);
        resolve(false);
      }
    });
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`🔄 Scheduling reconnection attempt ${this.reconnectAttempts + 1} in ${delay}ms`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      this.connect();
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
        } catch (error) {
          console.error('❌ Error sending heartbeat:', error);
        }
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    this.isManualClose = true;
    this.stopHeartbeat();
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      try {
        this.ws.close();
      } catch (error) {
        console.error('❌ Error closing WebSocket:', error);
      }
      this.ws = null;
    }
    
    this.notifyConnectionChange(false);
  }

  /**
   * Send message to WebSocket server
   */
  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (error) {
        console.error('❌ Error sending WebSocket message:', error);
      }
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send message');
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get user ID
   */
  getUserId(): string | null {
    return this.userId;
  }

  /**
   * Subscribe to analysis updates
   */
  onAnalysisUpdate(callback: MessageCallback): () => void {
    this.messageCallbacks.add(callback);
    return () => {
      this.messageCallbacks.delete(callback);
    };
  }

  /**
   * Subscribe to connection changes
   */
  onConnectionChange(callback: ConnectionCallback): () => void {
    this.connectionCallbacks.add(callback);
    // Immediately call with current status
    callback(this.isConnected());
    return () => {
      this.connectionCallbacks.delete(callback);
    };
  }

  /**
   * Notify all message callbacks
   */
  private notifyMessageReceived(data: AnalysisUpdate) {
    this.messageCallbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('❌ Error in message callback:', error);
      }
    });
  }

  /**
   * Notify all connection callbacks
   */
  private notifyConnectionChange(connected: boolean) {
    this.connectionCallbacks.forEach(callback => {
      try {
        callback(connected);
      } catch (error) {
        console.error('❌ Error in connection callback:', error);
      }
    });
  }
}

// Singleton instance
let websocketClient: WebSocketClient | null = null;

/**
 * Get WebSocket client instance (singleton)
 */
export function getWebSocketClient(): WebSocketClient {
  if (!websocketClient) {
    websocketClient = new WebSocketClient();
  }
  return websocketClient;
}

/**
 * React hook for WebSocket connection
 */
export function useWebSocket() {
  const clientRef = useRef<WebSocketClient>(getWebSocketClient());
  const [, forceUpdate] = useState({});

  // Force update on mount to ensure latest state
  useEffect(() => {
    forceUpdate({});
  }, []);

  return clientRef.current;
}
