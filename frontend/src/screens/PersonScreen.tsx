import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, TextInput, Button, ActivityIndicator } from 'react-native-paper';
import { useQueryClient, useQuery } from '@tanstack/react-query';
import { labelFaceCluster, getClusterPhotos } from '../services/api';
import { getPhotoImageURL } from '../utils/photoUtils';
import AppHeader from '../components/AppHeader';
import PhotoGrid from '../components/PhotoGrid';
import { API_BASE_URL, getPhoto } from '../services/api';

export default function PersonScreen({ navigation, route }: any) {
  const { cluster } = route.params;
  const [name, setName] = useState('');
  const [isLabeling, setIsLabeling] = useState(false);
  const queryClient = useQueryClient();

  // Fetch all photos for this cluster
  const { data: clusterPhotosData, isLoading: photosLoading, error: photosError } = useQuery({
    queryKey: ['clusterPhotos', cluster.cluster_id],
    queryFn: () => getClusterPhotos(cluster.cluster_id),
    retry: 1,
  });

  const onLabel = async () => {
    if (!name.trim()) return;
    
    setIsLabeling(true);
    try {
      await labelFaceCluster(cluster.cluster_id, name.trim());
      
      // Invalidate the clusters cache to refresh PeopleScreen
      queryClient.invalidateQueries({ queryKey: ['clusters'] });
      
      // Update the local state to reflect the change
      navigation.setParams({
        cluster: {
          ...cluster,
          label: name.trim(),
        },
      });
      
      // Clear the input field after successful labeling
      setName('');
      
      console.log(`Successfully labeled ${cluster.cluster_id} as "${name.trim()}"`);
    } catch (error) {
      console.error('Error labeling cluster:', error);
    } finally {
      setIsLabeling(false);
    }
  };

  // Function to get the URI for a photo
  const getPhotoUri = (photo: any) => {
    // For cluster photos from the new API, photo has path property
    if (photo && photo.path) {
      return getPhotoImageURL({ path: photo.path });
    }
    // For cluster photos, photo is just a string path (fallback)
    if (typeof photo === 'string') {
      return getPhotoImageURL({ path: photo });
    }
    // For regular photo objects
    return getPhotoImageURL(photo);
  };

  // Handle photo selection
  const handlePhotoPress = (photo: any) => {
    // Handle case where photo is from the new cluster photos API
    if (photo && photo.id && photo.path) {
      navigation.navigate('PhotoViewer', {
        photoId: photo.id,
        photo: {
          id: photo.id,
          filename: photo.filename,
          path: photo.path,
          photo_id: photo.id,
        },
      });
    }
    // Handle case where photo is just a string path (from clusters - fallback)
    else if (typeof photo === 'string') {
      // Extract filename from path
      const filename = photo.split(/[/\\]/).pop() || photo;

      navigation.navigate('PhotoViewer', {
        photoId: filename, // Use filename as ID for cluster photos
        photo: {
          id: filename,
          filename: filename,
          path: photo,
          photo_id: filename,
        },
      });
    } else {
      // Handle case where photo is an object (normal photos)
      navigation.navigate('PhotoViewer', {
        photoId: photo.photo_id || photo.id,
        photo: {
          id: photo.photo_id || photo.id,
          filename: photo.filename,
          path: photo.path,
          photo_id: photo.photo_id || photo.id,
        },
      });
    }
  };

  return (
    <View style={styles.container}>
      <AppHeader title={cluster.label || 'Person'} />

      <View style={styles.headerContainer}>
        <Text style={styles.photoCount}>Photos: {cluster.photo_count}</Text>
        {cluster.label && (
          <Text style={styles.currentLabel}>Current label: {cluster.label}</Text>
        )}
        <View style={styles.labelContainer}>
          <TextInput
            mode="outlined"
            label={cluster.label ? "Update person name" : "Label person"}
            value={name}
            onChangeText={setName}
            style={styles.input}
            disabled={isLabeling}
            theme={{
              colors: {
                background: '#FFFFFF',
                text: '#000000',
                placeholder: '#666666',
                primary: '#4285F4',
                outline: '#CCCCCC',
              }
            }}
            textColor="#000000"
          />
          <Button 
            mode="contained" 
            onPress={onLabel} 
            style={styles.button}
            loading={isLabeling}
            disabled={isLabeling || !name.trim()}
          >
            {isLabeling ? 'Saving...' : (cluster.label ? 'Update' : 'Save Label')}
          </Button>
        </View>
      </View>

      {photosLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator animating={true} size="large" />
          <Text style={styles.loadingText}>Loading photos...</Text>
        </View>
      ) : photosError ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>Error loading photos: {photosError.message}</Text>
        </View>
      ) : clusterPhotosData && clusterPhotosData.photos && clusterPhotosData.photos.length > 0 ? (
        <PhotoGrid
          data={clusterPhotosData.photos}
          getUri={getPhotoUri}
          onPress={handlePhotoPress}
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No photos found in this cluster</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  headerContainer: {
    padding: 12,
    backgroundColor: '#000000',
  },
  photoCount: {
    fontSize: 16,
    marginBottom: 8,
    color: '#FFFFFF',
  },
  currentLabel: {
    fontSize: 14,
    color: '#CCCCCC',
    marginBottom: 8,
    fontStyle: 'italic',
  },
  labelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  input: {
    flex: 1,
    marginRight: 8,
    backgroundColor: '#FFFFFF',
  },
  button: {
    paddingHorizontal: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  emptyText: {
    color: '#FFFFFF',
    fontSize: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  loadingText: {
    color: '#FFFFFF',
    fontSize: 16,
    marginTop: 16,
  },
});
