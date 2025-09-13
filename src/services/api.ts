import axios from 'axios';

export const API_BASE_URL = 'https://99bbe5275a1f.ngrok-free.app';

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 20000,
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
export const getAllPhotos = async (limit = 10000) => {
  try {
    console.log(`Fetching all photos with limit: ${limit}`);
    const response = await api.get(`/photos?limit=${limit}`);
    console.log('getAllPhotos response:', response.data);
    return response.data;
  } catch (error) {
    console.error('getAllPhotos error:', error);
    throw error;
  }
};

export const startIndexing = async (directory: string, recursive = true) =>
  (await api.post('/index', { directory, recursive })).data;
export const getTask = async (taskId: string) => (await api.get(`/tasks/${taskId}`)).data;

export const startFaceClustering = async () => (await api.post('/faces/cluster')).data;
export const getFaceClusters = async () => (await api.get('/faces/clusters')).data;
export const labelFaceCluster = async (cluster_id: string, name: string) =>
  (await api.post('/faces/label', { cluster_id, name })).data;

export const createGroup = async (group_name: string, cluster_ids: string[]) =>
  (await api.post('/groups/create', { group_name, cluster_ids })).data;
export const getGroups = async () => (await api.get('/groups')).data;

export const buildRelationships = async () => (await api.post('/relationships/build')).data;
export const getRelationships = async () => (await api.get('/relationships')).data;

