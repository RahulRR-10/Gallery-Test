
import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { ActivityIndicator, Button, FAB } from 'react-native-paper';
import { useQuery } from '@tanstack/react-query';
import { getFaceClusters, startFaceClustering, getTask } from '../services/api';
import AppHeader from '../components/AppHeader';
import FullscreenLoader from '../components/FullscreenLoader';
import PhotoGrid from '../components/PhotoGrid';
import { API_BASE_URL, getPhoto } from '../services/api';

export default function PeopleScreen({ navigation }: any) {
  const { data, isLoading, refetch } = useQuery({ queryKey: ['clusters'], queryFn: getFaceClusters });
  const clusters = data?.clusters || [];
  const [taskId, setTaskId] = React.useState<string | null>(null);
  const [taskStatus, setTaskStatus] = React.useState<any>(null);

  const onCluster = async () => {
    try {
      const res = await startFaceClustering();
      setTaskId(res.task_id);
    } catch {}
  };

  React.useEffect(() => {
    let t: any;
    if (taskId) {
      const poll = async () => {
        try {
          const s = await getTask(taskId);
          setTaskStatus(s);
          if (s.status !== 'completed' && s.status !== 'failed') {
            t = setTimeout(poll, 1500);
          } else if (s.status === 'completed') {
            refetch();
          }
        } catch {}
      };
      poll();
    }
    return () => t && clearTimeout(t);
  }, [taskId]);

  // Function to get the URI for a cluster's sample photo
  const getClusterPhotoUri = (item: any) => {
  if (item.cluster_id) {
    return item.sample_photos?.[0]?.filename
      ? `${API_BASE_URL}/images/${encodeURIComponent(item.sample_photos[0].filename)}`
      : '';
  } else {
    return item.filename
      ? `${API_BASE_URL}/images/${encodeURIComponent(item.filename)}`
      : '';
  }
};

  return (
    <View style={styles.container}>
      <AppHeader title="People" />
      <FullscreenLoader visible={!!taskId && taskStatus?.status === 'running'} title="Clustering faces" progress={taskStatus?.progress} />
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

