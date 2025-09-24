
import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { ActivityIndicator, Button, FAB } from 'react-native-paper';
import { useQuery } from '@tanstack/react-query';
import { useFocusEffect } from '@react-navigation/native';
import { getFaceClusters, startFaceClustering, getTask } from '../services/api';
import AppHeader from '../components/AppHeader';
import FullscreenLoader from '../components/FullscreenLoader';
import PhotoGrid from '../components/PhotoGrid';
import { API_BASE_URL, getPhoto } from '../services/api';

export default function PeopleScreen({ navigation }: any) {
  const { data, isLoading, refetch } = useQuery({ 
    queryKey: ['clusters'], 
    queryFn: getFaceClusters,
    staleTime: 0, // Always refetch when screen is focused
    refetchOnWindowFocus: true
  });
  const clusters = data?.clusters || [];
  const [taskId, setTaskId] = React.useState<string | null>(null);
  const [taskStatus, setTaskStatus] = React.useState<any>(null);

  console.log('PeopleScreen render - clusters:', clusters);
  console.log('PeopleScreen render - isLoading:', isLoading);

  // Refetch when screen comes into focus (e.g., when returning from PersonScreen)
  useFocusEffect(
    React.useCallback(() => {
      console.log('PeopleScreen focused - refetching data');
      refetch();
    }, [refetch])
  );

  const onCluster = async () => {
    try {
      const res = await startFaceClustering();
      console.log('Started clustering with task ID:', res.task_id);
      setTaskId(res.task_id);
    } catch (error) {
      console.error('Failed to start clustering:', error);
    }
  };

  const onManualRefresh = async () => {
    console.log('Manual refresh triggered');
    await refetch();
  };

  React.useEffect(() => {
    let t: any;
    if (taskId) {
      const poll = async () => {
        try {
          const s = await getTask(taskId);
          console.log('Task status:', s);
          setTaskStatus(s);
          if (s.status !== 'completed' && s.status !== 'failed') {
            t = setTimeout(poll, 1500);
          } else if (s.status === 'completed') {
            console.log('Clustering completed! Refreshing data...');
            // Clear task state and refetch data
            setTaskId(null);
            setTaskStatus(null);
            // Small delay to ensure backend is ready, then refetch
            setTimeout(() => {
              console.log('Refetching clusters after completion...');
              refetch();
            }, 500);
          } else if (s.status === 'failed') {
            console.log('Clustering failed:', s.error);
            // Clear task state on failure
            setTaskId(null);
            setTaskStatus(null);
          }
        } catch (error) {
          console.error('Error polling task status:', error);
        }
      };
      poll();
    }
    return () => t && clearTimeout(t);
  }, [taskId, refetch]);

  // Function to get the URI for a cluster's sample photo
  const getClusterPhotoUri = (item: any) => {
    console.log('getClusterPhotoUri called with:', item);
    
    if (item.cluster_id) {
      // This is a face cluster
      const samplePhotos = item.sample_photos || [];
      if (samplePhotos.length > 0) {
        // Handle both object format and string format
        const firstPhoto = samplePhotos[0];
        let filename = '';
        
        if (typeof firstPhoto === 'string') {
          // If it's a path string, extract filename
          filename = firstPhoto.split('\\').pop() || firstPhoto.split('/').pop() || firstPhoto;
        } else if (firstPhoto && firstPhoto.filename) {
          // If it's an object with filename
          filename = firstPhoto.filename;
        } else if (firstPhoto && firstPhoto.path) {
          // If it's an object with path, extract filename
          filename = firstPhoto.path.split('\\').pop() || firstPhoto.path.split('/').pop() || firstPhoto.path;
        }
        
        console.log('Cluster photo filename:', filename);
        return filename ? `${API_BASE_URL}/images/${encodeURIComponent(filename)}` : '';
      }
      return '';
    } else {
      // This is a regular photo
      return item.filename
        ? `${API_BASE_URL}/images/${encodeURIComponent(item.filename)}`
        : '';
    }
  };

  return (
    <View style={styles.container}>
      <AppHeader title="People" />
      <FullscreenLoader 
        visible={!!taskId && taskStatus?.status === 'running'} 
        title="Clustering faces" 
        progress={taskStatus?.progress}
        message={taskStatus?.message} 
      />
      {isLoading && <ActivityIndicator style={{ marginTop: 16 }} />}
      
      {clusters.length === 0 && !isLoading ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No face clusters found</Text>
          <Button 
            mode="contained" 
            onPress={onCluster}
            style={styles.button}
          >
            Cluster Faces
          </Button>
          <Button 
            mode="outlined" 
            onPress={onManualRefresh}
            style={styles.button}
          >
            Refresh
          </Button>
        </View>
      ) : (
        <>
          <PhotoGrid 
            data={clusters} 
            getUri={getClusterPhotoUri} 
            onPress={(cluster) => navigation.navigate('Person', { cluster })} 
          />
          
          <FAB
            style={styles.fab}
            icon="refresh"
            onPress={onManualRefresh}
            disabled={!!taskId && taskStatus?.status === 'running'}
          />
          
          <FAB
            style={[styles.fab, { bottom: 80 }]}
            icon="account-group"
            onPress={onCluster}
            disabled={!!taskId && taskStatus?.status === 'running'}
          />
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 18,
    marginBottom: 20,
  },
  button: {
    marginTop: 10,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
});

