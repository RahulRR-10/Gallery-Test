import React, { useState } from 'react';
import { FlatList, Image, TouchableOpacity, Dimensions, View, StyleSheet, Text } from 'react-native';
import { ActivityIndicator } from 'react-native-paper';

const numColumns = 3;
const screenWidth = Dimensions.get('window').width;
const spacing = 8; // Increased spacing between photos
const size = Math.floor((screenWidth - spacing * (numColumns + 1)) / numColumns);

type Props = {
  data: any[];
  getUri: (item: any) => string;
  onPress: (item: any) => void;
};

interface PhotoItemProps {
  item: any;
  getUri: (item: any) => string;
  onPress: (item: any) => void;
}

function PhotoItem({ item, getUri, onPress }: PhotoItemProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handleImageLoad = () => {
    setLoading(false);
  };

  const handleImageError = () => {
    setLoading(false);
    setError(true);
  };

  const formatDate = (timestamp: number) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString();
  };

  // Determine if this is a face cluster item
  const isCluster = item.cluster_id !== undefined;

  return (
    <TouchableOpacity 
      style={styles.photoContainer} 
      onPress={() => onPress(item)}
      activeOpacity={0.8}
    >
      <View style={styles.imageContainer}>
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#666" />
          </View>
        )}
        
        {error ? (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>📷</Text>
            <Text style={styles.errorLabel}>Failed to load</Text>
          </View>
        ) : (
          <Image 
            source={{ uri: getUri(item) }} 
            style={styles.image}
            onLoad={handleImageLoad}
            onError={handleImageError}
            resizeMode="cover"
          />
        )}
        
        {/* Photo metadata overlay */}
        <View style={styles.metadataOverlay}>
          {isCluster ? (
            // Face cluster metadata
            <>
              <Text style={styles.clusterLabel}>
                {item.label || `Cluster ${item.cluster_id}`}
              </Text>
              {item.photo_count && (
                <Text style={styles.photoCountText}>
                  {item.photo_count} photos
                </Text>
              )}
            </>
          ) : (
            // Regular photo metadata
            <>
              {item.timestamp && (
                <Text style={styles.dateText}>{formatDate(item.timestamp)}</Text>
              )}
              {item.objects && item.objects.length > 0 && (
                <View style={styles.objectsContainer}>
                  <Text style={styles.objectsText}>
                    {item.objects.slice(0, 2).join(', ')}
                    {item.objects.length > 2 && '...'}
                  </Text>
                </View>
              )}
            </>
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
}

export default function PhotoGrid({ data, getUri, onPress }: Props) {
  return (
    <FlatList
      data={data}
      keyExtractor={(item) => String(item.id)}
      numColumns={numColumns}
      contentContainerStyle={styles.gridContainer}
      columnWrapperStyle={numColumns > 1 ? styles.row : undefined}
      showsVerticalScrollIndicator={false}
      renderItem={({ item }) => (
        <PhotoItem 
          item={item} 
          getUri={getUri} 
          onPress={onPress} 
        />
      )}
    />
  );
}

const styles = StyleSheet.create({
  gridContainer: {
    padding: spacing,
    paddingBottom: spacing * 2, // Extra bottom padding for better scrolling
  },
  row: {
    justifyContent: 'space-between',
    marginBottom: spacing,
    paddingHorizontal: spacing / 2,
  },
  photoContainer: {
    width: size,
    height: size,
    marginBottom: spacing,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#f0f0f0',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  imageContainer: {
    flex: 1,
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  errorText: {
    fontSize: 24,
    marginBottom: 4,
  },
  errorLabel: {
    fontSize: 10,
    color: '#666',
    textAlign: 'center',
  },
  metadataOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 6,
    paddingVertical: 4,
  },
  dateText: {
    color: 'white',
    fontSize: 10,
    fontWeight: '500',
  },
  objectsContainer: {
    marginTop: 2,
  },
  objectsText: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 9,
    fontStyle: 'italic',
  },
  // Cluster-specific styles
  clusterLabel: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  photoCountText: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 10,
    marginTop: 2,
  },
});






