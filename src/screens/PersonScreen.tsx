import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, TextInput, Button } from 'react-native-paper';
import { labelFaceCluster } from '../services/api';
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
          label: name
        }
      });
    } catch (error) {
      console.error('Error labeling cluster:', error);
    }
  };

  // Function to get the URI for a photo
  const getPhotoUri = (photo: any) => {
    return `${API_BASE_URL}/images/${photo.filename}`;
  };

  // Handle photo selection
  const handlePhotoPress = (photo: any) => {
    // Check if photo is a string or an object
    
    navigation.navigate('PhotoViewer', { 
      photoId: photo.photo_id,
      photo: { 
        id: photo.photo_id, 
        filename: typeof photo === 'string' ? photo : photo.filename,
        path: typeof photo === 'string' ? photo : photo.path,
        photo_id: photo.photo_id
      }
    });
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
          <Button 
            mode="contained" 
            onPress={onLabel}
            style={styles.button}
          >
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
    backgroundColor: '#fff'
  },
  headerContainer: {
    padding: 12
  },
  photoCount: {
    fontSize: 16,
    marginBottom: 8
  },
  labelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12
  },
  input: {
    flex: 1,
    marginRight: 8
  },
  button: {
    paddingHorizontal: 8
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  }
});

