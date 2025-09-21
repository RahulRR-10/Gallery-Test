import React, { useState, useEffect } from 'react';
import {
  FlatList,
  Image,
  TouchableOpacity,
  View,
  Dimensions,
  Alert,
  StyleSheet,
  SectionList,
  Text as RNText,
} from 'react-native';
import { ActivityIndicator, Button, Text, FAB } from 'react-native-paper';
import { useMutation, useQuery } from '@tanstack/react-query';
import { searchPhotos, getAllPhotos, API_BASE_URL } from '../services/api';
import { getPhotoImageURL } from '../utils/photoUtils';
import AppHeader from '../components/AppHeader';
import PhotoGrid from '../components/PhotoGrid';
import IndexingManager from '../components/IndexingManager';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

const numColumns = 4; // More photos per row like native gallery
const { width } = Dimensions.get('window');
const photoSize = (width - 8) / numColumns; // 2px gap between photos

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

interface PhotoSection {
  title: string;
  data: any[];
}

// Group photos by date
const groupPhotosByDate = (photos: any[]): PhotoSection[] => {
  if (!photos || photos.length === 0) return [];
  
  // Sort photos by timestamp (newest first)
  const sortedPhotos = [...photos].sort((a, b) => {
    const timestampA = a.exif_timestamp || a.timestamp || 0;
    const timestampB = b.exif_timestamp || b.timestamp || 0;
    return timestampB - timestampA;
  });
  
  const groups: { [key: string]: any[] } = {};
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  sortedPhotos.forEach(photo => {
    const timestamp = (photo.exif_timestamp || photo.timestamp || 0) * 1000;
    const photoDate = new Date(timestamp);
    
    let dateKey: string;
    
    if (photoDate.toDateString() === today.toDateString()) {
      dateKey = 'Today';
    } else if (photoDate.toDateString() === yesterday.toDateString()) {
      dateKey = 'Yesterday';
    } else {
      // Format as "Month Day, Year" or "Month Day" if same year
      const isSameYear = photoDate.getFullYear() === today.getFullYear();
      dateKey = photoDate.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: isSameYear ? undefined : 'numeric'
      });
    }
    
    if (!groups[dateKey]) {
      groups[dateKey] = [];
    }
    groups[dateKey].push(photo);
  });
  
  return Object.entries(groups).map(([title, data]) => ({ title, data }));
};

export default function GalleryScreen({ navigation, route }: Props) {
  const initial = route.params?.initialQuery ?? null;
  const [photos, setPhotos] = useState<any[]>([]);
  const [showIndexingManager, setShowIndexingManager] = useState(false);
  const [photoSections, setPhotoSections] = useState<PhotoSection[]>([]);

  // Use React Query to automatically fetch photos
  const { data: allPhotosData, isLoading, error, refetch } = useQuery({
    queryKey: ['allPhotos'],
    queryFn: () => getAllPhotos(),
    retry: 1,
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
      const sections = groupPhotosByDate((allPhotosData as any).results);
      setPhotoSections(sections);
    } else if (error && !isLoading) {
      // Show indexing manager if no photos found
      setShowIndexingManager(true);
    }
  }, [initial, allPhotosData, error, isLoading]);

  useEffect(() => {
    if (mutation.data?.results) {
      setPhotos(mutation.data.results);
      const sections = groupPhotosByDate(mutation.data.results);
      setPhotoSections(sections);
    }
  }, [mutation.data]);

  const handleIndexingComplete = (indexedPhotos: any[]) => {
    setPhotos(indexedPhotos);
    const sections = groupPhotosByDate(indexedPhotos);
    setPhotoSections(sections);
    setShowIndexingManager(false);
    refetch(); // Refresh the query
    Alert.alert(
      'Success',
      `Loaded ${indexedPhotos.length} photos into gallery!`,
    );
  };

  const handleIndexingError = (error: string) => {
    Alert.alert('Error', error);
    setShowIndexingManager(false);
  };

  const handleRefresh = () => {
    refetch();
  };

  const handleLoadAllPhotos = () => {
    refetch();
  };

  // Render individual photo item
  const renderPhotoItem = ({ item }: { item: any }) => (
    <TouchableOpacity
      style={styles.photoItem}
      onPress={() => navigation.navigate('PhotoViewer', { 
        photoId: item.id, 
        photo: item 
      })}
    >
      <Image
        source={{ uri: getPhotoImageURL(item) }}
        style={styles.photoImage}
        resizeMode="cover"
      />
    </TouchableOpacity>
  );

  // Render section with photos in grid
  const renderSection = ({ item: section }: { item: PhotoSection }) => (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>{section.title}</Text>
      </View>
      <FlatList
        data={section.data}
        renderItem={renderPhotoItem}
        numColumns={numColumns}
        scrollEnabled={false}
        contentContainerStyle={styles.photosGrid}
        columnWrapperStyle={numColumns > 1 ? styles.row : undefined}
        keyExtractor={(item) => item.id}
      />
    </View>
  );

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
          <SectionList
            sections={photoSections}
            renderItem={renderPhotoItem}
            renderSectionHeader={({ section }) => (
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>{section.title}</Text>
              </View>
            )}
            keyExtractor={(item) => item.id}
            showsVerticalScrollIndicator={false}
            stickySectionHeadersEnabled={true}
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
    paddingHorizontal: 2,
  },
  row: {
    justifyContent: 'space-around',
  },
  photoItem: {
    width: photoSize,
    height: photoSize,
    margin: 1,
  },
  photoImage: {
    width: '100%',
    height: '100%',
  },
});
