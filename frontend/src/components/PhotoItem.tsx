import React, { memo, useState } from 'react';
import {
  View,
  Image,
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { getPhotoImageURL, formatPhotoTimestamp } from '../utils/photoUtils';

interface PhotoItemProps {
  photo: any;
  onPress: (photo: any) => void;
  itemSize: number;
}

const PhotoItem: React.FC<PhotoItemProps> = memo(({ photo, onPress, itemSize }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handlePress = () => {
    onPress(photo);
  };

  const handleImageLoad = () => {
    setLoading(false);
  };

  const handleImageError = () => {
    setLoading(false);
    setError(true);
  };

  const imageUrl = getPhotoImageURL(photo);

  return (
    <TouchableOpacity 
      style={[styles.container, { width: itemSize, height: itemSize }]}
      onPress={handlePress}
      activeOpacity={0.8}
    >
      <View style={styles.imageContainer}>
        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#007AFF" />
          </View>
        )}
        
        {error ? (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Failed to load</Text>
          </View>
        ) : (
          <Image
            source={{ uri: imageUrl }}
            style={styles.image}
            resizeMode="cover"
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        )}
        
        {/* Photo overlay with timestamp */}
        {photo.timestamp && (
          <View style={styles.overlay}>
            <Text style={styles.timestamp}>
              {formatPhotoTimestamp(photo.timestamp)}
            </Text>
          </View>
        )}
        
        {/* Objects indicator */}
        {photo.objects && photo.objects.length > 0 && (
          <View style={styles.objectsIndicator}>
            <Text style={styles.objectsCount}>
              {photo.objects.length}
            </Text>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for memo optimization
  return (
    prevProps.photo.id === nextProps.photo.id &&
    prevProps.itemSize === nextProps.itemSize
  );
});

const styles = StyleSheet.create({
  container: {
    margin: 2,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    overflow: 'hidden',
  },
  imageContainer: {
    flex: 1,
    position: 'relative',
  },
  image: {
    flex: 1,
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
    zIndex: 1,
  },
  errorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  errorText: {
    fontSize: 10,
    color: '#888',
    textAlign: 'center',
  },
  overlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  timestamp: {
    color: 'white',
    fontSize: 10,
    fontWeight: '500',
  },
  objectsIndicator: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: '#007AFF',
    borderRadius: 10,
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  objectsCount: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
});

export default PhotoItem;