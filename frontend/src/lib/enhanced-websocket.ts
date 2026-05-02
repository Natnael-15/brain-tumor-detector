/**
 * Enhanced WebSocket Client with Advanced Features
 * Hospital-grade WebSocket connection with multi-URL fallback, reconnection, and health monitoring
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

/**
 * Enhanced WebSocket Client with Multi-URL Fallback
 * Implements hospital-grade reliability features:
 * - Multi-URL fallback (localhost → 127.0.0.1 → 0.0.0.0)
 * - Exponential backoff reconnection
 * - Heartbeat/keep-alive mechanism
 * - Automatic connection recovery
 */
class EnhancedWebSocketClient {
  private ws: WebSocket | null = null;
  
  // Connection URLs with fallback strategy (configurable via environment or defaults)
  private primaryUrl = this.getDefaultWsUrl();
  private fallbackUrls = typeof window !== 'undefined' && process.env.NEXT_PUBLIC_WS_FALLBACK_URLS
    ? process.env.NEXT_PUBLIC_WS_FALLBACK_URLS.split(',')
    : [this.getDefaultWsUrl('127.0.0.1:8000'), this.getDefaultWsUrl('0.0.0.0:8000')];
  private currentUrlIndex = -1; // Start at -1 so first call gets primary URL

  private getDefaultWsUrl(hostPort = 'localhost:8000'): string {
    if (typeof window === 'undefined') return `ws://${hostPort}`;
    
    // If we have an explicit env var, use it
    if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
    
    // Check if the hostPort itself is a local address
    const isLocalHost = hostPort.includes('localhost') || hostPort.includes('127.0.0.1') || hostPort.includes('0.0.0.0');
    
    // If we are connecting to a local server, we usually want ws: (insecure)
    // even if the page is https:, because localhost doesn't have valid SSL.
    const protocol = isLocalHost ? 'ws:' : (window.location.protocol === 'https:' ? 'wss:' : 'ws:');
    
    return `${protocol}//${hostPort}`;
  }
  
  // Reconnection settings
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 15;
  private baseReconnectDelay = 1000;
  private maxReconnectDelay = 10000;
  private connectionTimeout = 5000; // Connection timeout in milliseconds
  
  // State management
  private messageCallbacks: Set<MessageCallback> = new Set();
  private connectionCallbacks: Set<ConnectionCallback> = new Set();
  private userId: string | null = null;
  private isManualClose = false;
  
  // Health monitoring
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private heartbeatIntervalMs = 25000; // 25 seconds
  private reconnectTimeout: NodeJS.Timeout | null = null;
  
  /**
   * Get next URL to try (cycles through primary and fallback URLs)
   */
  private getNextUrl(): string {
    if (this.currentUrlIndex === -1) {
      // First attempt - use primary URL
      this.currentUrlIndex = 0;
      return this.primaryUrl;
    }
    
    // Cycle through fallback URLs
    if (this.currentUrlIndex < this.fallbackUrls.length) {
      const url = this.fallbackUrls[this.currentUrlIndex];
      this.currentUrlIndex++;
      return url;
    }
    
    // Reset to primary URL after trying all fallbacks
    this.currentUrlIndex = 0;
    return this.primaryUrl;
  }

  /**
   * Calculate exponential backoff delay with cap
   */
  private getReconnectDelay(): number {
    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    return delay;
  }

  /**
   * Connect to WebSocket server with multi-URL fallback
   */
  async connect(userId?: string): Promise<boolean> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('✅ EnhancedWebSocket already connected');
      return true;
    }

    this.isManualClose = false;
    this.userId = userId || this.userId || `user_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
    
    return new Promise((resolve) => {
      try {
        const currentUrl = this.getNextUrl();
        const wsUrl = `${currentUrl}/ws/${this.userId}`;
        console.log(`🔗 EnhancedWebSocket attempting connection: ${wsUrl} (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log(`✅ EnhancedWebSocket connected successfully to ${currentUrl}`);
          this.reconnectAttempts = 0;
          this.currentUrlIndex = -1; // Reset for next connection cycle
          this.notifyConnectionChange(true);
          this.startHeartbeat();
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('📨 EnhancedWebSocket message received:', data);
            
            // Handle different message types
            if (data.type === 'connection_established') {
              console.log('✅ Connection established:', data);
            } else if (data.type === 'analysis_update') {
              this.notifyMessageReceived(data);
            } else if (data.type === 'pong') {
              console.log('💓 Heartbeat pong received');
            } else {
              // Forward all messages to callbacks
              this.notifyMessageReceived(data);
            }
          } catch (error) {
            console.error('❌ Error parsing EnhancedWebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error(`❌ EnhancedWebSocket error on ${currentUrl}:`, error);
          resolve(false);
        };

        this.ws.onclose = () => {
          console.log(`🔴 EnhancedWebSocket closed from ${currentUrl}`);
          this.stopHeartbeat();
          this.notifyConnectionChange(false);
          
          if (!this.isManualClose && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('❌ Max reconnection attempts reached. Connection failed.');
          }
        };

        // Timeout if connection takes too long
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.warn(`⚠️ EnhancedWebSocket connection timeout for ${currentUrl}`);
            this.ws?.close();
            resolve(false);
          }
        }, this.connectionTimeout);

      } catch (error) {
        console.error('❌ Error creating EnhancedWebSocket:', error);
        resolve(false);
      }
    });
  }

  /**
   * Schedule reconnection attempt with exponential backoff
   */
  private scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = this.getReconnectDelay();
    console.log(`🔄 Scheduling EnhancedWebSocket reconnection attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts} in ${delay}ms`);
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`🔄 EnhancedWebSocket reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      this.connect();
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat() {
    this.stopHeartbeat();
    
    console.log(`💓 Starting EnhancedWebSocket heartbeat (every ${this.heartbeatIntervalMs / 1000}s)`);
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ 
            type: 'ping', 
            timestamp: new Date().toISOString() 
          }));
          console.log('💓 Heartbeat ping sent');
        } catch (error) {
          console.error('❌ Error sending heartbeat:', error);
        }
      }
    }, this.heartbeatIntervalMs);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
      console.log('💓 EnhancedWebSocket heartbeat stopped');
    }
  }

  /**
   * Disconnect from WebSocket (manual close)
   */
  disconnect() {
    console.log('🔴 EnhancedWebSocket manual disconnect');
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
        console.error('❌ Error closing EnhancedWebSocket:', error);
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
        console.error('❌ Error sending EnhancedWebSocket message:', error);
      }
    } else {
      console.warn('⚠️ EnhancedWebSocket not connected, cannot send message');
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
let enhancedWebSocketClient: EnhancedWebSocketClient | null = null;

/**
 * Get Enhanced WebSocket client instance (singleton)
 */
export function getEnhancedWebSocketClient(): EnhancedWebSocketClient {
  if (!enhancedWebSocketClient) {
    enhancedWebSocketClient = new EnhancedWebSocketClient();
    console.log('🚀 EnhancedWebSocket client initialized');
  }
  return enhancedWebSocketClient;
}

/**
 * React hook for Enhanced WebSocket connection
 */
export function useEnhancedWebSocket() {
  const clientRef = useRef<EnhancedWebSocketClient>(getEnhancedWebSocketClient());
  const [isConnected, setIsConnected] = useState(clientRef.current.isConnected());

  useEffect(() => {
    // Subscribe to connection state changes
    const unsubscribe = clientRef.current.onConnectionChange((connected) => {
      setIsConnected(connected);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return clientRef.current;
}
