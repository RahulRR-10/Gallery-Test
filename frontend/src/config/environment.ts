/**
 * Environment Configuration for API Connectivity
 * Handles dynamic API base URL configuration for different environments
 */

import { Platform } from 'react-native';
import { LOCAL_IP } from '@env';

/**
 * Configuration options for different environments
 */
const Config = {
  development: {
    // Local development - use your computer's IP address from environment variables
    localIP: LOCAL_IP || 'your-ip-address', // Fallback to default IP
    localPort: 8000,
    
    // Alternative: ngrok URL for external access
    ngrokURL: 'your-ngrok-url',
    
    timeout: 60000, // Increased to 60 seconds
  },
  production: {
    // Production server URL
    baseURL: 'https://your-production-api.com',
    timeout: 30000, // Increased timeout for production too
  },
};

/**
 * Get the appropriate API base URL
 */
export const getAPIBaseURL = (): string => {
  const isDevelopment = __DEV__;
  
  if (isDevelopment) {
    const { localIP, localPort, ngrokURL } = Config.development;
    
    // Priority order:
    // 1. Use ngrok URL if available (for external access)
    // 2. Fall back to local IP (for same network access)
    
    if (ngrokURL && ngrokURL !== 'your-ngrok-url') {
      // Use ngrok if you've updated it with a real URL
      return ngrokURL;
    } else {
      // Use local IP address
      return `http://${localIP}:${localPort}`;
    }
  } else {
    // Production environment
    return Config.production.baseURL;
  }
};

/**
 * Get timeout value for current environment
 */
export const getTimeout = (): number => {
  const isDevelopment = __DEV__;
  return isDevelopment ? Config.development.timeout : Config.production.timeout;
};

/**
 * Update local IP when it changes
 */
export const updateLocalIP = (newIP: string): void => {
  Config.development.localIP = newIP;
};

/**
 * Update ngrok URL when you get a new tunnel
 */
export const updateNgrokURL = (newURL: string): void => {
  Config.development.ngrokURL = newURL;
};

/**
 * Current API configuration
 */
export const API_CONFIG = {
  baseURL: getAPIBaseURL(),
  timeout: getTimeout(),
};

export default Config;
