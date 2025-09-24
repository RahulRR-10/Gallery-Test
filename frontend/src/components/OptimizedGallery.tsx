import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  Dimensions,
  RefreshControl,
  Text,
  ActivityIndicator,
} from 'react-native';
import { useInfiniteQuery } from '@tanstack/react-query';
import { getPhotosBatch } from '../services/api';
import { throttle } from '../utils/photoUtils';
import PhotoItem from './PhotoItem';

const { width } = Dimensions.get('window');
const ITEM_MARGIN = 4;
const ITEMS_PER_ROW = 3;
const ITEM_SIZE = (width - (ITEMS_PER_ROW + 1) * ITEM_MARGIN) / ITEMS_PER_ROW;
const PHOTOS_PER_PAGE = 30;

interface OptimizedGalleryProps {
  onPhotoPress?: (photo: any) => void;
}

const OptimizedGallery: React.FC<OptimizedGalleryProps> = ({ 
  onPhotoPress = () => {} 
}) => {
  const [refreshing, setRefreshing] = useState(false);

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
    refetch
  } = useInfiniteQuery({
    queryKey: ['photos-paginated'],
    queryFn: ({ pageParam = 0 }) => 
      getPhotosBatch(PHOTOS_PER_PAGE, pageParam * PHOTOS_PER_PAGE),
    getNextPageParam: (lastPage, allPages) => {
      if (lastPage?.pagination?.has_more) {
        return allPages.length;
      }
      return undefined;
    },
    initialPageParam: 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });

  // Flatten all photos from all pages
  const allPhotos = useMemo(() => {
    if (!data?.pages) return [];
    
    return data.pages.reduce((acc, page) => {
      if (page?.results) {
        return [...acc, ...page.results];
      }
      return acc;
    }, [] as any[]);
  }, [data?.pages]);

  // Throttled load more function
  const throttledLoadMore = useMemo(
    () => throttle(() => {
      if (hasNextPage && !isFetchingNextPage) {
        fetchNextPage();
      }
    }, 1000),
    [hasNextPage, isFetchingNextPage, fetchNextPage]
  );

  const handleLoadMore = useCallback(() => {
    throttledLoadMore();
  }, [throttledLoadMore]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await refetch();
    } finally {
      setRefreshing(false);
    }
  }, [refetch]);

  const renderPhoto = useCallback(({ item }: { item: any }) => (
    <PhotoItem 
      photo={item} 
      onPress={onPhotoPress} 
      itemSize={ITEM_SIZE}
    />
  ), [onPhotoPress]);

  const renderFooter = useCallback(() => {
    if (isFetchingNextPage) {
      return (
        <View style={styles.footer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading more photos...</Text>
        </View>
      );
    }
    
    if (!hasNextPage && allPhotos.length > 0) {
      return (
        <View style={styles.footer}>
          <Text style={styles.endText}>
            All {allPhotos.length} photos loaded
          </Text>
        </View>
      );
    }
    
    return null;
  }, [isFetchingNextPage, hasNextPage, allPhotos.length]);

  const keyExtractor = useCallback((item: any) => item.id, []);

  if (isLoading && !refreshing) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading your photos...</Text>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>
          Failed to load photos: {error?.message || 'Unknown error'}
        </Text>
      </View>
    );
  }

  if (allPhotos.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.emptyText}>No photos found</Text>
        <Text style={styles.emptySubtext}>
          Try indexing some photos first
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>
          {allPhotos.length} Photos
        </Text>
        {hasNextPage && (
          <Text style={styles.headerSubtext}>
            Loading...
          </Text>
        )}
      </View>
      
      <FlatList
        data={allPhotos}
        renderItem={renderPhoto}
        keyExtractor={keyExtractor}
        numColumns={ITEMS_PER_ROW}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        maxToRenderPerBatch={15}
        windowSize={10}
        initialNumToRender={15}
        removeClippedSubviews={true}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor="#007AFF"
          />
        }
        ListFooterComponent={renderFooter}
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#f8f8f8',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  headerSubtext: {
    fontSize: 14,
    color: '#666',
  },
  listContent: {
    padding: ITEM_MARGIN / 2,
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 8,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  endText: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  errorText: {
    fontSize: 16,
    color: '#FF3B30',
    textAlign: 'center',
    marginBottom: 8,
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
});

export default OptimizedGallery;