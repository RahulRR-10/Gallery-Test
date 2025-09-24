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
  SectionList,
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

interface PhotoSection {
  title: string;
  subtitle: string;
  data: Photo[];
}



const PicturesScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [photoSections, setPhotoSections] = useState<PhotoSection[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchMode, setSearchMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Photo[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  const groupPhotosByDate = (photos: Photo[]): PhotoSection[] => {
    // Sort photos by timestamp (newest first)
    const sortedPhotos = [...photos].sort((a, b) => b.timestamp - a.timestamp);
    
    const groups: { [key: string]: Photo[] } = {};
    
    sortedPhotos.forEach(photo => {
      const date = new Date(photo.timestamp * 1000);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      
      let dateKey: string;
      let subtitle: string;
      
      if (date.toDateString() === today.toDateString()) {
        dateKey = 'Today';
        subtitle = date.toLocaleDateString('en-US', { 
          weekday: 'long', 
          month: 'long', 
          day: 'numeric' 
        });
      } else if (date.toDateString() === yesterday.toDateString()) {
        dateKey = 'Yesterday';
        subtitle = date.toLocaleDateString('en-US', { 
          weekday: 'long', 
          month: 'long', 
          day: 'numeric' 
        });
      } else if (date.getFullYear() === today.getFullYear()) {
        dateKey = date.toLocaleDateString('en-US', { 
          month: 'long', 
          day: 'numeric' 
        });
        subtitle = date.toLocaleDateString('en-US', { 
          weekday: 'long' 
        });
      } else {
        dateKey = date.toLocaleDateString('en-US', { 
          year: 'numeric', 
          month: 'long', 
          day: 'numeric' 
        });
        subtitle = date.toLocaleDateString('en-US', { 
          weekday: 'long' 
        });
      }
      
      const fullKey = `${dateKey}|${subtitle}`;
      if (!groups[fullKey]) {
        groups[fullKey] = [];
      }
      groups[fullKey].push(photo);
    });
    
    return Object.entries(groups).map(([key, data]) => {
      const [title, subtitle] = key.split('|');
      return {
        title,
        subtitle,
        data,
      };
    });
  };

  const fetchPhotos = async () => {
    try {
      const response = await fetch(`${API_CONFIG.baseURL}/api/photos`);
      const data = await response.json();
      console.log('Photos API response:', data);
      const photosData = data.results || [];
      setPhotos(photosData);
      setPhotoSections(groupPhotosByDate(photosData));
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
          // Pass the complete photo object
          (navigation as any).navigate('PhotoViewer', { photo: item });
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

  const renderSectionHeader = ({ section }: { section: PhotoSection }) => (
    <View style={[styles.sectionHeader, { backgroundColor: theme.colors.background }]}>
      <Text style={[styles.sectionTitle, { color: theme.colors.onBackground }]}>
        {section.title}
      </Text>
      <Text style={[styles.sectionSubtitle, { color: theme.colors.onSurfaceVariant }]}>
        {section.subtitle} • {section.data.length} photo{section.data.length !== 1 ? 's' : ''}
      </Text>
    </View>
  );

  const renderSectionData = ({ item, index, section }: { item: Photo; index: number; section: PhotoSection }) => {
    // Only render every 3rd item to create rows
    if (index % 3 !== 0) return null;
    
    // Get the current row of photos (up to 3)
    const rowPhotos = section.data.slice(index, index + 3);
    
    return (
      <View style={styles.photoRow}>
        {rowPhotos.map((photo) => (
          <View key={photo.id} style={styles.photoContainer}>
            {renderPhoto({ item: photo })}
          </View>
        ))}
        {/* Fill empty spaces if the row has less than 3 photos */}
        {Array(3 - rowPhotos.length).fill(null).map((_, emptyIndex) => (
          <View key={`empty-${index}-${emptyIndex}`} style={styles.photoContainer} />
        ))}
      </View>
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
      {searchMode ? (
        <FlatList
          data={searchResults}
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
                name="magnify"
                size={64}
                color={theme.colors.onSurfaceVariant}
              />
              <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                No search results found
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
      ) : (
        <SectionList
          sections={photoSections}
          renderItem={renderSectionData}
          renderSectionHeader={renderSectionHeader}
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
          stickySectionHeadersEnabled={true}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <MaterialCommunityIcons
                name="image-outline"
                size={64}
                color={theme.colors.onSurfaceVariant}
              />
              <Text style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                No photos found
              </Text>
            </View>
          }
        />
      )}
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
    width: ITEM_SIZE,
    height: ITEM_SIZE,
    marginHorizontal: 2,
  },
  photoImage: {
    width: ITEM_SIZE,
    height: ITEM_SIZE,
    borderRadius: 8,
  },
  photoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
    paddingHorizontal: 16,
  },
  sectionHeader: {
    width: '100%',
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
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