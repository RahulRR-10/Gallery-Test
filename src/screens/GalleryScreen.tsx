import React, { useState } from 'react';
import { FlatList, Image, TouchableOpacity, View, Dimensions, Alert, StyleSheet } from 'react-native';
import { ActivityIndicator, Button, Text } from 'react-native-paper';
import { useMutation } from '@tanstack/react-query';
import { searchPhotos, getAllPhotos, API_BASE_URL } from '../services/api';
import AppHeader from '../components/AppHeader';
import PhotoGrid from '../components/PhotoGrid';
import IndexingManager from '../components/IndexingManager';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

const numColumns = 3;
const size = Math.floor(Dimensions.get('window').width / numColumns);

type RootStackParamList = {
  Root: undefined;
  PhotoViewer: { photoId: string; photo: any };
  Person: { cluster: any };
};

type RootTabParamList = {
  Gallery: { initialQuery?: string | null } | undefined;
  Search: undefined;
  People: undefined;
  Groups: undefined;
  Relationships: undefined;
  Settings: undefined;
};

type Props = NativeStackScreenProps<RootStackParamList, 'Root'> & {
  navigation: any; // Tab navigation
  route: any; // Tab route
};

export default function GalleryScreen({ navigation, route }: Props) {
  const initial = route.params?.initialQuery ?? null;
  const [photos, setPhotos] = useState<any[]>([]);
  const [showIndexingManager, setShowIndexingManager] = useState(false);
  const [isLoadingAll, setIsLoadingAll] = useState(false);
  
  const mutation = useMutation({ mutationFn: (payload: any) => searchPhotos(payload) });
  const loadAllMutation = useMutation({ mutationFn: () => getAllPhotos() });

  React.useEffect(() => {
    if (initial) {
      // If there's a search query, use the search API
      mutation.mutate({ query: initial, limit: 60 });
    } else {
      // If no query, try to load all photos first
      loadAllPhotos();
    }
  }, [initial]);

  React.useEffect(() => {
    if (mutation.data?.results) {
      setPhotos(mutation.data.results);
    }
  }, [mutation.data]);

  const loadAllPhotos = async () => {
    try {
      console.log('Loading all photos...');
      setIsLoadingAll(true);
      const response = await getAllPhotos();
      console.log('Load all photos response:', response);
      if (response.results && response.results.length > 0) {
        setPhotos(response.results);
        console.log(`Loaded ${response.results.length} photos`);
      } else {
        console.log('No photos found, showing indexing manager');
        // No photos found, show indexing manager
        setShowIndexingManager(true);
      }
    } catch (error) {
      console.error('Load all photos error:', error);
      // If error, show indexing manager
      setShowIndexingManager(true);
    } finally {
      setIsLoadingAll(false);
    }
  };

  const handleIndexingComplete = (indexedPhotos: any[]) => {
    setPhotos(indexedPhotos);
    setShowIndexingManager(false);
    Alert.alert('Success', `Loaded ${indexedPhotos.length} photos into gallery!`);
  };

  const handleIndexingError = (error: string) => {
    Alert.alert('Error', error);
    setShowIndexingManager(false);
  };

  const handleLoadAllPhotos = () => {
    loadAllPhotos();
  };

  if (showIndexingManager) {
    return (
      <View style={{ flex: 1 }}>
        <AppHeader title="Gallery Setup" />
        <IndexingManager 
          onIndexingComplete={handleIndexingComplete}
          onError={handleIndexingError}
        />
      </View>
    );
  }

  return (
    <View style={{ flex: 1 }}>
      <AppHeader title="Gallery" />
      
      {photos.length === 0 && !mutation.isLoading && !isLoadingAll && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>
            No photos found
          </Text>
          <Text style={styles.emptySubtitle}>
            Start indexing your photos to load them into the gallery
          </Text>
          <Button 
            mode="contained"
            onPress={() => setShowIndexingManager(true)}
            style={styles.indexButton}
          >
            Start Indexing
          </Button>
        </View>
      )}

      {(mutation.isLoading || isLoadingAll) && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" />
          <Text style={styles.loadingText}>Loading photos...</Text>
        </View>
      )}

      {photos.length > 0 && (
        <>
          <View style={styles.headerContainer}>
            <View style={styles.headerLeft}>
              <Text style={styles.photoCount}>
                {photos.length} photos
              </Text>
              <Text style={styles.lastUpdated}>
                Last updated: {new Date().toLocaleTimeString()}
              </Text>
            </View>
            <Button 
              mode="outlined"
              onPress={handleLoadAllPhotos}
              compact
              style={styles.loadAllButton}
              icon="refresh"
            >
              Refresh
            </Button>
          </View>
          <PhotoGrid
            data={photos}
            getUri={(item) => `${API_BASE_URL}/images/${encodeURIComponent(item.filename)}`}
            onPress={(item) => navigation.navigate('PhotoViewer', { photoId: item.id, photo: item })}
          />
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyTitle: {
    fontSize: 18,
    marginBottom: 16,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  emptySubtitle: {
    marginBottom: 20,
    textAlign: 'center',
    color: '#666',
  },
  indexButton: {
    marginTop: 8,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
  },
  headerContainer: {
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  headerLeft: {
    flex: 1,
  },
  photoCount: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  lastUpdated: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  loadAllButton: {
    minWidth: 100,
  },
});

