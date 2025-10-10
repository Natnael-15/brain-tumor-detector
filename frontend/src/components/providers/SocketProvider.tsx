'use client';

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface SocketContextType {
  isConnected: boolean;
  emit: (event: string, data: any) => void;
  on: (event: string, callback: (data: any) => void) => void;
  off: (event: string, callback: (data: any) => void) => void;
}

const SocketContext = createContext<SocketContextType | undefined>(undefined);

export function useSocket() {
  const context = useContext(SocketContext);
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
}

interface SocketProviderProps {
  children: ReactNode;
}

export function SocketProvider({ children }: SocketProviderProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<any>(null);

  useEffect(() => {
    // Mock socket connection for now
    // In production, this would use socket.io-client
    const mockSocket = {
      emit: (event: string, data: any) => {
        console.log('Socket emit:', event, data);
      },
      on: (event: string, callback: (data: any) => void) => {
        console.log('Socket on:', event);
      },
      off: (event: string, callback: (data: any) => void) => {
        console.log('Socket off:', event);
      },
    };

    setSocket(mockSocket);
    setIsConnected(true);

    return () => {
      setIsConnected(false);
      setSocket(null);
    };
  }, []);

  const value: SocketContextType = {
    isConnected,
    emit: socket?.emit || (() => {}),
    on: socket?.on || (() => {}),
    off: socket?.off || (() => {}),
  };

  return <SocketContext.Provider value={value}>{children}</SocketContext.Provider>;
}