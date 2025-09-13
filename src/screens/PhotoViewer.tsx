import React, { useState } from 'react';
import { Image, ScrollView, View, StyleSheet, Dimensions, TouchableOpacity } from 'react-native';
import { Chip, List, Text, Card, Divider, ActivityIndicator } from 'react-native-paper';
import { API_BASE_URL, getPhoto } from '../services/api';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import AppHeader from '../components/AppHeader';

const screenWidth = Dimensions.get('window').width;
const imageHeight = screenWidth * 0.75; // 4:3 aspect ratio

type RootStackParamList = {
  Root: undefined;
  PhotoViewer: { photoId: string; photo: any };
  Person: { cluster: any };
};

type Props = NativeStackScreenProps<RootStackParamList, 'PhotoViewer'>;

export default function PhotoViewer({ navigation, route }: Props) {
  const { photo } = route.params;
  const [details, setDetails] = React.useState<any>(photo);
  const [imageLoading, setImageLoading] = useState(true);
  const [imageError, setImageError] = useState(false);

  React.useEffect(() => {
    // Fetch details by id to ensure fresh data
    getPhoto(photo.id || photo.photo_id).then(setDetails).catch(() => {});
  }, [photo.id, photo.photo_id]);

  // Use the exact path field from API response
  const uri = `${API_BASE_URL}/images/${encodeURIComponent(photo.filename)}`;

  const formatDate = (timestamp: number) => {
    if (!timestamp) return 'Unknown date';
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const formatFileSize = (bytes: number) => {
    if (!bytes) return 'Unknown size';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <View style={styles.container}>
      <AppHeader title={photo.filename} />
      
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* Main Image */}
        <Card style={styles.imageCard} elevation={3}>
          <View style={styles.imageContainer}>
            {imageLoading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#666" />
                <Text style={styles.loadingText}>Loading image...</Text>
              </View>
            )}
            
            {imageError ? (
              <View style={styles.errorContainer}>
                <Text style={styles.errorIcon}>📷</Text>
                <Text style={styles.errorText}>Failed to load image</Text>
              </View>
            ) : (
              <Image 
                source={{ uri }} 
                style={styles.image}
                onLoad={() => setImageLoading(false)}
                onError={() => {
                  setImageLoading(false);
                  setImageError(true);
                }}
                resizeMode="contain"
              />
            )}
          </View>
        </Card>

        {/* Photo Information */}
        <Card style={styles.infoCard} elevation={2}>
          <Card.Content>
            <Text style={styles.sectionTitle}>Photo Information</Text>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Filename:</Text>
              <Text style={styles.infoValue}>{photo.filename}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Date:</Text>
              <Text style={styles.infoValue}>{formatDate(photo.timestamp)}</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Path:</Text>
              <Text style={styles.infoValue} numberOfLines={2}>{details.path}</Text>
            </View>
          </Card.Content>
        </Card>

        {/* Detected Objects */}
        {Array.isArray(details.objects) && details.objects.length > 0 && (
          <Card style={styles.infoCard} elevation={2}>
            <Card.Content>
              <Text style={styles.sectionTitle}>Detected Objects</Text>
              <View style={styles.chipContainer}>
                {details.objects.map((obj: string, idx: number) => (
                  <Chip key={idx} style={styles.chip} textStyle={styles.chipText}>
                    {obj}
                  </Chip>
                ))}
              </View>
            </Card.Content>
          </Card>
        )}

        {/* Faces Information */}
        {details.faces && details.faces.length > 0 && (
          <Card style={styles.infoCard} elevation={2}>
            <Card.Content>
              <Text style={styles.sectionTitle}>Detected Faces</Text>
              {details.faces.map((face: any, i: number) => (
                <View key={i} style={styles.faceItem}>
                  <Text style={styles.faceTitle}>
                    Face {i + 1}
                  </Text>
                  <Text style={styles.faceDetails}>
                    Cluster: {face.cluster_id || 'Unknown'}
                  </Text>
                  {face.confidence && (
                    <Text style={styles.faceDetails}>
                      Confidence: {(face.confidence * 100).toFixed(1)}%
                    </Text>
                  )}
                  {i < details.faces.length - 1 && <Divider style={styles.divider} />}
                </View>
              ))}
            </Card.Content>
          </Card>
        )}

        {/* Relationships */}
        {details.relationships && details.relationships.length > 0 && (
          <Card style={styles.infoCard} elevation={2}>
            <Card.Content>
              <Text style={styles.sectionTitle}>Relationships</Text>
              {details.relationships.map((rel: any, i: number) => (
                <View key={i} style={styles.relationshipItem}>
                  <Text style={styles.relationshipText}>
                    {rel.person1} → {rel.person2} ({rel.relationship_type})
                  </Text>
                  {i < details.relationships.length - 1 && <Divider style={styles.divider} />}
                </View>
              ))}
            </Card.Content>
          </Card>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  imageCard: {
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  imageContainer: {
    height: imageHeight,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loadingContainer: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 8,
    color: '#666',
    fontSize: 14,
  },
  errorContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  errorText: {
    color: '#666',
    fontSize: 16,
  },
  infoCard: {
    marginBottom: 16,
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: 8,
    flexWrap: 'wrap',
  },
  infoLabel: {
    fontWeight: '600',
    color: '#666',
    minWidth: 80,
    marginRight: 8,
  },
  infoValue: {
    flex: 1,
    color: '#333',
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    backgroundColor: '#e3f2fd',
    marginBottom: 4,
  },
  chipText: {
    color: '#1976d2',
    fontWeight: '500',
  },
  faceItem: {
    paddingVertical: 8,
  },
  faceTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  faceDetails: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  relationshipItem: {
    paddingVertical: 8,
  },
  relationshipText: {
    fontSize: 14,
    color: '#333',
  },
  divider: {
    marginTop: 8,
    backgroundColor: '#e0e0e0',
  },
});

