import axios from 'axios';
import { API_CONFIG } from '../config/environment';

export const API_BASE_URL = API_CONFIG.baseURL;

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: API_CONFIG.timeout,
});

export const getStatus = async () => (await api.get('/status')).data;
export const getStats = async () => (await api.get('/stats')).data;

export const searchPhotos = async (payload: {
  query?: string | null;
  person?: string | null;
  group?: string | null;
  relationship?: string | null;
  time_filter?: string | null;
  limit?: number;
}) => (await api.post('/search', payload)).data;

export const getPhoto = async (id: number) => (await api.get(`/photos/${id}`)).data;
export const getAllPhotos = async (limit = 1000, offset = 0) => {
  try {
    console.log(`Fetching photos with limit: ${limit}, offset: ${offset}`);
    const response = await api.get(`/photos?limit=${limit}&offset=${offset}`);
    console.log('getAllPhotos response:', response.data);
    return response.data;
  } catch (error) {
    console.error('getAllPhotos error:', error);
    throw error;
  }
};

export const getPhotosBatch = async (limit = 20, offset = 0) => {
  try {
    const response = await api.get(`/photos?limit=${limit}&offset=${offset}`);
    return response.data;
  } catch (error) {
    console.error('getPhotosBatch error:', error);
    throw error;
  }
};

export const startIndexing = async (directory: string, recursive = true) =>
  (await api.post('/index', { directory, recursive })).data;
export const getTask = async (taskId: string) => {
  try {
    const response = await api.get(`/tasks/${taskId}`);
    console.log('Task status response:', response.data);
    return response.data;
  } catch (error) {
    console.error('getTask error:', error);
    throw error;
  }
};

export const startFaceClustering = async () => (await api.post('/faces/cluster')).data;

export const getFaceClusters = async () => {
  try {
    const response = await api.get('/faces/clusters');
    console.log('getFaceClusters response:', response.data);
    return response.data;
  } catch (error) {
    console.error('getFaceClusters error:', error);
    throw error;
  }
};
export const labelFaceCluster = async (cluster_id: string, name: string) =>
  (await api.post('/faces/label', { cluster_id, name })).data;

export const getClusterPhotos = async (cluster_id: string, limit: number = 1000) => {
  try {
    const response = await api.get(`/faces/clusters/${cluster_id}/photos?limit=${limit}`);
    console.log(`getClusterPhotos response for cluster ${cluster_id}:`, response.data);
    return response.data;
  } catch (error) {
    console.error('getClusterPhotos error:', error);
    throw error;
  }
};

export const createGroup = async (group_name: string, cluster_ids: string[]) =>
  (await api.post('/groups/create', { group_name, cluster_ids })).data;
export const getGroups = async () => (await api.get('/groups')).data;

export const buildRelationships = async () => (await api.post('/relationships/build')).data;
export const getRelationships = async () => (await api.get('/relationships')).data;

// Auto-indexing APIs
export const startAutoIndexing = async () => (await api.post('/auto-index/start')).data;
export const stopAutoIndexing = async () => (await api.post('/auto-index/stop')).data;
export const getAutoIndexingStatus = async () => (await api.get('/auto-index/status')).data;

