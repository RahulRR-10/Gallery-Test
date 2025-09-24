import React, { useState, useEffect } from 'react';
import {
  View,
  Alert,
  StyleSheet,
  FlatList,
  Text,
  ActivityIndicator,
} from 'react-native';
import { FAB } from 'react-native-paper';
import { useMutation } from '@tanstack/react-query';
import { searchPhotos } from '../services/api';
import AppHeader from '../components/AppHeader';
import IndexingManager from '../components/IndexingManager';
import OptimizedGallery from '../components/OptimizedGallery';
import PhotoItem from '../components/PhotoItem';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

type RootStackParamList = {
  Root: undefined;
  PhotoViewer: { photoId: string; photo: any };
  Person: { cluster: any };
};

type Props = NativeStackScreenProps<RootStackParamList, 'Root'> & {
  navigation: any;
  route: any;
};

export default function GalleryScreen({ navigation, route }: Props) {
  const initialQuery = route.params?.initialQuery ?? null;
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showIndexingManager, setShowIndexingManager] = useState(false);
  const [isSearchMode, setIsSearchMode] = useState(!!initialQuery);

  const searchMutation = useMutation({
    mutationFn: (payload: any) => searchPhotos(payload),
    onSuccess: (data) => {
      if (data?.results) {
        setSearchResults(data.results);
        setIsSearchMode(true);
      }
    },
    onError: (error) => {
      console.error('Search error:', error);
      Alert.alert('Search Error', 'Failed to search photos');
    }
  });

  useEffect(() => {
    if (initialQuery) {
      searchMutation.mutate({ query: initialQuery, limit: 100 });
    }
  }, [initialQuery, searchMutation]);

  const handleIndexingComplete = (indexedPhotos: any[]) => {
    setShowIndexingManager(false);
    Alert.alert(
      'Success', 
      `Indexed ${indexedPhotos.length} photos successfully!`,
      [{ text: 'OK' }]
    );
  };

  const handleIndexingError = (error: string) => {
    Alert.alert('Indexing Error', error);
    setShowIndexingManager(false);
  };

  const handlePhotoPress = (photo: any) => {
    navigation.navigate('PhotoViewer', {
      photoId: photo.id,
      photo: photo,
    });
  };

  const handleBackToGallery = () => {
    setIsSearchMode(false);
    setSearchResults([]);
  };

  if (showIndexingManager) {
    return (
      <View style={styles.container}>
        <AppHeader title="Gallery" />
        <IndexingManager
          onIndexingComplete={handleIndexingComplete}
          onError={handleIndexingError}
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader 
        title={isSearchMode ? "Search Results" : "Gallery"} 
        onBack={isSearchMode ? handleBackToGallery : undefined}
      />
      
      {isSearchMode ? (
        <SearchResultsView 
          photos={searchResults}
          onPhotoPress={handlePhotoPress}
          isLoading={searchMutation.isPending}
        />
      ) : (
        <OptimizedGallery onPhotoPress={handlePhotoPress} />
      )}
      
      <FAB
        style={styles.fab}
        icon="folder-plus"
        onPress={() => setShowIndexingManager(true)}
        label="Add Photos"
      />
    </View>
  );
}

interface SearchResultsViewProps {
  photos: any[];
  onPhotoPress: (photo: any) => void;
  isLoading: boolean;
}

const SearchResultsView: React.FC<SearchResultsViewProps> = ({ 
  photos, 
  onPhotoPress, 
  isLoading 
}) => {
  const renderPhoto = ({ item }: { item: any }) => (
    <PhotoItem 
      photo={item} 
      onPress={onPhotoPress} 
      itemSize={120}
    />
  );

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Searching photos...</Text>
      </View>
    );
  }

  if (photos.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.emptyText}>No photos found</Text>
        <Text style={styles.emptySubtext}>Try a different search term</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.searchHeader}>
        <Text style={styles.resultsText}>
          {photos.length} result{photos.length !== 1 ? 's' : ''}
        </Text>
      </View>
      <FlatList
        data={photos}
        renderItem={renderPhoto}
        keyExtractor={(item) => item.id}
        numColumns={3}
        contentContainerStyle={styles.searchResults}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  searchHeader: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#f8f8f8',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  resultsText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  searchResults: {
    padding: 8,
  },
  loadingText: {
    marginTop: 8,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#007AFF',
  },
});