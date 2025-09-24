import React, { useState, useEffect } from 'react';
import {
  FlatList,
  Image,
  TouchableOpacity,
  View,
  Dimensions,
  Alert,
  StyleSheet,
  RefreshControl,
} from 'react-native';
import { ActivityIndicator, Button, Text } from 'react-native-paper';
import { useMutation, useQuery } from '@tanstack/react-query';
import { searchPhotos, getAllPhotos, getAutoIndexingStatus, startAutoIndexing } from '../services/api';
import { getPhotoImageURL } from '../utils/photoUtils';
import AppHeader from '../components/AppHeader';
import IndexingManager from '../components/IndexingManager';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

const numColumns = 3; // 3 photos per row like mobile gallery
const { width } = Dimensions.get('window');
const spacing = 4; // Gap between photos
const photoSize = (width - (spacing * (numColumns + 1))) / numColumns;

type RootStackParamList = {
  Root: undefined;
  PhotoViewer: { photoId: string; photo: any };
  Person: { cluster: any };
};

type Props = NativeStackScreenProps<RootStackParamList, 'Root'> & {
  navigation: any; // Tab navigation
  route: any; // Tab route
};

export default function GalleryScreen({ navigation, route }: Props) {
  const initial = route.params?.initialQuery ?? null;
  const [photos, setPhotos] = useState<any[]>([]);
  const [showIndexingManager, setShowIndexingManager] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [autoIndexingEnabled, setAutoIndexingEnabled] = useState(false);

  // Use React Query to automatically fetch photos
  const { data: allPhotosData, isLoading, error, refetch } = useQuery({
    queryKey: ['allPhotos'],
    queryFn: () => getAllPhotos(),
    retry: 1,
  });

  // Check auto-indexing status
  const { data: autoIndexStatus } = useQuery({
    queryKey: ['autoIndexStatus'],
    queryFn: getAutoIndexingStatus,
    refetchInterval: 5000, // Check every 5 seconds
  });

  const mutation = useMutation({
    mutationFn: (payload: any) => searchPhotos(payload),
  });

  useEffect(() => {
    if (initial) {
      // If there's a search query, use the search API
      mutation.mutate({ query: initial, limit: 100 });
    } else if (allPhotosData?.results) {
      // Load all photos from React Query
      setPhotos((allPhotosData as any).results);
    } else if (error && !isLoading) {
      // Show indexing manager if no photos found
      setShowIndexingManager(true);
    }
  }, [initial, allPhotosData, error, isLoading, mutation]);

  useEffect(() => {
    if (mutation.data?.results) {
      setPhotos(mutation.data.results);
    }
  }, [mutation.data]);

  useEffect(() => {
    if (autoIndexStatus?.status) {
      setAutoIndexingEnabled(autoIndexStatus.status.running);
    }
  }, [autoIndexStatus]);

  const handleIndexingComplete = (indexedPhotos: any[]) => {
    setPhotos(indexedPhotos);
    setShowIndexingManager(false);
    refetch(); // Refresh the query
    Alert.alert(
      'Success',
      `Loaded ${indexedPhotos.length} photos into gallery!`,
    );
  };

  const handleIndexingError = (indexingError: string) => {
    Alert.alert('Error', indexingError);
    setShowIndexingManager(false);
  };

  const handleLoadAllPhotos = () => {
    refetch();
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refetch();
    } finally {
      setRefreshing(false);
    }
  };

  const handleEnableAutoIndexing = async () => {
    try {
      await startAutoIndexing();
      setAutoIndexingEnabled(true);
      Alert.alert('Success', 'Auto-indexing enabled! New photos will be automatically processed.');
    } catch (error) {
      Alert.alert('Error', 'Failed to enable auto-indexing');
    }
  };

  // Render individual photo item with loading state
  const renderPhotoItem = ({ item }: { item: any }) => (
    <TouchableOpacity
      style={styles.photoItem}
      onPress={() => navigation.navigate('PhotoViewer', { 
        photoId: item.id, 
        photo: item 
      })}
      activeOpacity={0.8}
    >
      <Image
        source={{ uri: getPhotoImageURL(item) }}
        style={styles.photoImage}
        resizeMode="cover"
      />
    </TouchableOpacity>
  );

  if (showIndexingManager) {
    return (
      <View style={styles.container}>
        <AppHeader title="Gallery Setup" />
        <IndexingManager
          onIndexingComplete={handleIndexingComplete}
          onError={handleIndexingError}
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="Gallery" />

      {photos.length === 0 && !isLoading && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyTitle}>No photos found</Text>
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

      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" />
          <Text style={styles.loadingText}>Loading photos...</Text>
        </View>
      )}

      {photos.length > 0 && (
        <>
          <View style={styles.headerContainer}>
            <View style={styles.headerLeft}>
              <Text style={styles.photoCount}>{photos.length} photos</Text>
              <Text style={styles.lastUpdated}>
                Last updated: {new Date().toLocaleTimeString()}
              </Text>
              <Text style={[styles.autoIndexStatus, { color: autoIndexingEnabled ? '#4CAF50' : '#FF9800' }]}>
                Auto-indexing: {autoIndexingEnabled ? '✅ Enabled' : '⚠️ Disabled'}
              </Text>
            </View>
            <View style={styles.headerButtons}>
              <Button
                mode="outlined"
                onPress={handleLoadAllPhotos}
                compact
                style={styles.actionButton}
                icon="refresh"
              >
                Refresh
              </Button>
              {!autoIndexingEnabled && (
                <Button
                  mode="contained"
                  onPress={handleEnableAutoIndexing}
                  compact
                  style={styles.actionButton}
                  icon="folder-sync"
                >
                  Enable Auto-Index
                </Button>
              )}
            </View>
          </View>
          <FlatList
            data={photos}
            renderItem={renderPhotoItem}
            numColumns={numColumns}
            keyExtractor={(item) => item.id}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={styles.photosGrid}
            columnWrapperStyle={numColumns > 1 ? styles.row : undefined}
            getItemLayout={(data, index) => ({
              length: photoSize + spacing,
              offset: (photoSize + spacing) * Math.floor(index / numColumns),
              index,
            })}
            initialNumToRender={15}
            maxToRenderPerBatch={15}
            windowSize={10}
            removeClippedSubviews={true}
            updateCellsBatchingPeriod={50}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={handleRefresh}
                tintColor="#6200ee"
                colors={['#6200ee']}
              />
            }
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
  autoIndexStatus: {
    fontSize: 11,
    marginTop: 2,
    fontWeight: '500',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    minWidth: 100,
  },
  loadAllButton: {
    minWidth: 100,
  },
  section: {
    paddingBottom: 16,
  },
  sectionHeader: {
    backgroundColor: '#f8f9fa',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: '#e9ecef',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#495057',
  },
  photosGrid: {
    paddingHorizontal: spacing / 2,
    paddingVertical: spacing / 2,
  },
  row: {
    justifyContent: 'space-between',
    paddingHorizontal: spacing / 2,
  },
  photoItem: {
    width: photoSize,
    height: photoSize,
    marginBottom: spacing,
    borderRadius: 4,
    overflow: 'hidden',
    backgroundColor: '#f5f5f5',
  },
  photoImage: {
    width: '100%',
    height: '100%',
    borderRadius: 4,
  },
});
