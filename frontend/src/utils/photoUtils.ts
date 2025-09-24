/**
 * Utility functions for handling photo paths and URLs
 */

import { API_BASE_URL } from '../services/api';

/**
 * Convert photo path to image URL
 * Handles different path formats from the backend
 */
export const getPhotoImageURL = (photo: any): string => {
  // Handle different possible fields
  const path = photo.path || photo.filename || photo.name || '';
  
  if (!path) {
    console.warn('No path found for photo:', photo);
    return '';
  }

  // Extract filename from full path
  let filename = '';
  
  if (path.includes('sample_photos')) {
    // If path contains sample_photos, extract everything after it
    const parts = path.split('sample_photos');
    if (parts.length > 1) {
      filename = parts[1].replace(/^[/\\]/, ''); // Remove leading slash/backslash
    } else {
      filename = path.split(/[/\\]/).pop() || '';
    }
  } else {
    // Otherwise, just use the filename
    filename = path.split(/[/\\]/).pop() || '';
  }

  // Clean up the filename
  filename = filename.replace(/\\/g, '/'); // Convert backslashes to forward slashes
  
  // Construct the URL
  const imageURL = `${API_BASE_URL}/images/${encodeURIComponent(filename)}`;
  
  console.log('Photo path conversion:', { 
    originalPath: path, 
    extractedFilename: filename, 
    finalURL: imageURL 
  });
  
  return imageURL;
};

/**
 * Extract just the filename from a photo object
 */
export const getPhotoFilename = (photo: any): string => {
  const path = photo.path || photo.filename || photo.name || '';
  return path.split(/[/\\]/).pop() || 'Unknown';
};

/**
 * Get relative path for display purposes
 */
export const getPhotoDisplayPath = (photo: any): string => {
  const path = photo.path || photo.filename || photo.name || '';
  
  if (path.includes('sample_photos')) {
    const parts = path.split('sample_photos');
    if (parts.length > 1) {
      return 'sample_photos' + parts[1];
    }
  }
  
  return path;
};

/**
 * Performance optimization utilities
 */

export interface PhotoSection {
  title: string;
  data: any[];
}

export const groupPhotosByDate = (photos: any[]): PhotoSection[] => {
  const grouped: { [key: string]: any[] } = {};
  
  photos.forEach(photo => {
    if (photo.timestamp) {
      // Create date string for grouping
      const date = new Date(photo.timestamp * 1000);
      const dateKey = date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
      
      if (!grouped[dateKey]) {
        grouped[dateKey] = [];
      }
      grouped[dateKey].push(photo);
    }
  });
  
  // Convert to array and sort by date (newest first)
  return Object.entries(grouped)
    .map(([title, data]) => ({ title, data }))
    .sort((a, b) => {
      const dateA = new Date(a.data[0].timestamp * 1000);
      const dateB = new Date(b.data[0].timestamp * 1000);
      return dateB.getTime() - dateA.getTime();
    });
};

export const formatPhotoTimestamp = (timestamp: number): string => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

// Debounce function for performance optimization
export const debounce = (func: Function, wait: number) => {
  let timeout: NodeJS.Timeout;
  return function executedFunction(...args: any[]) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Throttle function for scroll events
export const throttle = (func: Function, limit: number) => {
  let inThrottle: boolean;
  return function(this: any, ...args: any[]) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};