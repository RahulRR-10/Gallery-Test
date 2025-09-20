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