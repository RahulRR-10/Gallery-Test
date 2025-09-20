import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, TextInput, Button } from 'react-native-paper';
import { labelFaceCluster } from '../services/api';
import { getPhotoImageURL } from '../utils/photoUtils';
import AppHeader from '../components/AppHeader';
import PhotoGrid from '../components/PhotoGrid';
import { API_BASE_URL, getPhoto } from '../services/api';

export default function PersonScreen({ navigation, route }: any) {
  const { cluster } = route.params;
  const [name, setName] = useState('');

  const onLabel = async () => {
    if (!name) return;
    try {
      await labelFaceCluster(cluster.cluster_id, name);
      // Update the local state to reflect the change
      navigation.setParams({
        cluster: {
          ...cluster,
          label: name,
        },
      });
    } catch (error) {
      console.error('Error labeling cluster:', error);
    }
  };

  // Function to get the URI for a photo
  const getPhotoUri = (photo: any) => {
    // For cluster photos, photo is just a string path
    if (typeof photo === 'string') {
      return getPhotoImageURL({ path: photo });
    }
    // For regular photo objects
    return getPhotoImageURL(photo);
  };

  // Handle photo selection
  const handlePhotoPress = (photo: any) => {
    // Handle case where photo is just a string path (from clusters)
    if (typeof photo === 'string') {
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
      <AppHeader title={cluster.label || cluster.cluster_id || 'Person'} />

      <View style={styles.headerContainer}>
        <Text style={styles.photoCount}>Photos: {cluster.photo_count}</Text>
        <View style={styles.labelContainer}>
          <TextInput
            mode="outlined"
            label="Label person"
            value={name}
            onChangeText={setName}
            style={styles.input}
          />
          <Button mode="contained" onPress={onLabel} style={styles.button}>
            Save Label
          </Button>
        </View>
      </View>

      {cluster.sample_photos && cluster.sample_photos.length > 0 ? (
        <PhotoGrid
          data={cluster.sample_photos}
          getUri={getPhotoUri}
          onPress={handlePhotoPress}
        />
      ) : (
        <View style={styles.emptyContainer}>
          <Text>No photos found in this cluster</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  headerContainer: {
    padding: 12,
  },
  photoCount: {
    fontSize: 16,
    marginBottom: 8,
  },
  labelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  input: {
    flex: 1,
    marginRight: 8,
  },
  button: {
    paddingHorizontal: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
