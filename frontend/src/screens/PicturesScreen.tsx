import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  RefreshControl,
  TouchableOpacity,
  TextInput,
  Dimensions,
  Image,
} from 'react-native';
import { Text, useTheme, ActivityIndicator, Surface } from 'react-native-paper';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { useNavigation } from '@react-navigation/native';
import { API_CONFIG } from '../config/environment';

const { width } = Dimensions.get('window');
const ITEM_SIZE = (width - 48) / 3; // 3 columns with padding

interface Photo {
  id: string;
  filename: string;
  path: string;
  timestamp: number;
  similarity_score?: number;
  objects?: any[];
  faces?: any[];
  relationships?: any[];
}



const PicturesScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchMode, setSearchMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Photo[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  const fetchPhotos = async () => {
    try {
      const response = await fetch(`${API_CONFIG.baseURL}/api/photos`);
      const data = await response.json();
      console.log('Photos API response:', data);
      setPhotos(data.results || []);
    } catch (error) {
      console.error('Error fetching photos:', error);
      setPhotos([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setSearchLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.baseURL}/api/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Error searching photos:', error);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const toggleSearchMode = () => {
    try {
      if (searchMode) {
        // Exit search mode
        setSearchMode(false);
        setSearchQuery('');
        setSearchResults([]);
      } else {
        // Enter search mode
        setSearchMode(true);
      }
    } catch (error) {
      console.error('Error toggling search mode:', error);
    }
  };

  const renderPhoto = ({ item }: { item: Photo }) => {
    // Convert the backend path to the correct URL format
    // Backend returns "sample_photos\filename.jpg" but serves as "/images/filename.jpg"
    const imageUrl = `${API_CONFIG.baseURL}/images/${item.filename}`;
    
    return (
      <TouchableOpacity
        style={styles.photoContainer}
        onPress={() => {
          // Simple navigation without type checking for now
          (navigation as any).navigate('PhotoViewer', { photoId: item.id });
        }}
      >
        <Image
          source={{ uri: imageUrl }}
          style={styles.photoImage}
          resizeMode="cover"
        />
      </TouchableOpacity>
    );
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchPhotos();
  }, []);

  useEffect(() => {
    fetchPhotos();
  }, []);

  useEffect(() => {
    if (searchQuery) {
      const debounceTimer = setTimeout(() => {
        handleSearch(searchQuery);
      }, 500);
      return () => clearTimeout(debounceTimer);
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);





  if (loading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.loadingText, { color: theme.colors.onBackground }]}>
          Loading photos...
        </Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {/* Header */}
      <Surface style={[styles.header, { backgroundColor: theme.colors.surface }]}>
        <View style={styles.headerContent}>
          <Text style={[styles.headerTitle, { color: theme.colors.onSurface }]}>
            {photos.length} photos
          </Text>
          <Text style={[styles.headerSubtitle, { color: theme.colors.onSurfaceVariant }]}>
            Last updated: {new Date().toLocaleTimeString('en-US', { 
              hour: 'numeric', 
              minute: '2-digit',
              hour12: true 
            })}
          </Text>
        </View>
        
        <TouchableOpacity
          style={styles.searchButton}
          onPress={toggleSearchMode}
        >
          <MaterialCommunityIcons
            name={searchMode ? "close" : "magnify"}
            size={24}
            color={theme.colors.onSurface}
          />
        </TouchableOpacity>
      </Surface>

      {/* Search Input */}
      {searchMode && (
        <View
          style={[
            styles.searchContainer,
            { backgroundColor: theme.colors.surface },
          ]}
        >
          <TextInput
            style={[styles.searchInput, { color: theme.colors.onSurface }]}
            placeholder="Search everything... (e.g., 'John birthday cake last month')"
            placeholderTextColor={theme.colors.onSurfaceVariant}
            value={searchQuery}
            onChangeText={setSearchQuery}
            autoFocus={searchMode}
          />
        </View>
      )}

      {/* Photo Grid */}
      <FlatList
        data={searchMode ? searchResults : photos}
        renderItem={renderPhoto}
        numColumns={3}
        contentContainerStyle={styles.gridContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.colors.primary}
            colors={[theme.colors.primary]}
          />
        }
        showsVerticalScrollIndicator={false}
        removeClippedSubviews={true}
        maxToRenderPerBatch={50}
        windowSize={10}
        initialNumToRender={30}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <MaterialCommunityIcons
              name={searchMode ? "magnify" : "image-outline"}
              size={64}
              color={theme.colors.onSurfaceVariant}
            />
            <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
              {searchMode ? 'No search results found' : 'No photos found'}
            </Text>
          </View>
        }
        ListFooterComponent={
          searchLoading ? (
            <View style={styles.searchLoadingContainer}>
              <ActivityIndicator size="large" color={theme.colors.primary} />
              <Text style={[styles.searchLoadingText, { color: theme.colors.onBackground }]}>
                Searching...
              </Text>
            </View>
          ) : null
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    elevation: 2,
  },
  headerContent: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
  },
  headerSubtitle: {
    fontSize: 12,
    marginTop: 2,
    opacity: 0.7,
  },
  searchButton: {
    padding: 8,
  },
  searchContainer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    justifyContent: 'center',
  },
  searchInput: {
    fontSize: 16,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  gridContent: {
    padding: 16,
  },
  photoContainer: {
    flex: 1,
    margin: 2,
    maxWidth: ITEM_SIZE,
  },
  photoImage: {
    width: ITEM_SIZE,
    height: ITEM_SIZE,
    borderRadius: 8,
  },
  sectionHeader: {
    width: '100%',
    paddingVertical: 16,
    paddingHorizontal: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  sectionSubtitle: {
    fontSize: 12,
    marginTop: 2,
    opacity: 0.7,
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
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 16,
    marginTop: 16,
    textAlign: 'center',
  },
  searchLoadingContainer: {
    paddingVertical: 20,
    alignItems: 'center',
  },
  searchLoadingText: {
    marginTop: 8,
    fontSize: 14,
  },
});

export default PicturesScreen;