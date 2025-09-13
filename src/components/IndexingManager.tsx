import React, { useState, useEffect } from 'react';
import { View, Text, Alert, StyleSheet } from 'react-native';
import { ActivityIndicator, ProgressBar, Card, Button } from 'react-native-paper';
import { startIndexing, getTask, getAllPhotos } from '../services/api';

interface IndexingManagerProps {
  onIndexingComplete: (photos: any[]) => void;
  onError: (error: string) => void;
}

export default function IndexingManager({ onIndexingComplete, onError }: IndexingManagerProps) {
  const [isIndexing, setIsIndexing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [totalFiles, setTotalFiles] = useState(0);
  const [processedFiles, setProcessedFiles] = useState(0);

  const startPhotoIndexing = async (directory: string = 'sample_photos') => {
    try {
      console.log('Starting photo indexing...');
      setIsIndexing(true);
      setProgress(0);
      setStatus('Starting indexing...');
      setProcessedFiles(0);
      setTotalFiles(0);

      // Start indexing
      const response = await startIndexing(directory, true);
      console.log('Indexing response:', response);
      setTaskId(response.task_id);
      setStatus('Indexing photos...');

      // Start polling for progress
      pollTaskStatus(response.task_id);
    } catch (error) {
      console.error('Indexing error:', error);
      onError('Failed to start indexing: ' + (error as Error).message);
      setIsIndexing(false);
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    try {
      const response = await getTask(taskId);
      
      if (response.status === 'completed') {
        setStatus('Indexing completed! Loading photos...');
        setProgress(100);
        
        // Load all photos after indexing is complete
        await loadAllPhotos();
      } else if (response.status === 'failed') {
        onError('Indexing failed: ' + (response.error || 'Unknown error'));
        setIsIndexing(false);
      } else if (response.status === 'running') {
        setProgress(response.progress || 0);
        setTotalFiles(response.total || 0);
        setProcessedFiles(Math.floor((response.progress || 0) * (response.total || 0) / 100));
        setStatus(`Processing photos... ${processedFiles}/${totalFiles}`);
        
        // Continue polling
        setTimeout(() => pollTaskStatus(taskId), 1000);
      }
    } catch (error) {
      console.error('Task status error:', error);
      onError('Failed to check indexing status: ' + (error as Error).message);
      setIsIndexing(false);
    }
  };

  const loadAllPhotos = async () => {
    try {
      setStatus('Loading all photos into gallery...');
      const response = await getAllPhotos();
      
      if (response.results && response.results.length > 0) {
        setStatus(`Loaded ${response.results.length} photos successfully!`);
        onIndexingComplete(response.results);
      } else {
        onError('No photos found after indexing');
      }
    } catch (error) {
      console.error('Load photos error:', error);
      onError('Failed to load photos: ' + (error as Error).message);
    } finally {
      setIsIndexing(false);
    }
  };

  const cancelIndexing = () => {
    setIsIndexing(false);
    setStatus('Indexing cancelled');
    setProgress(0);
    setTaskId(null);
  };

  if (!isIndexing) {
    return (
      <Card style={styles.card}>
        <Text style={styles.title}>
          Photo Indexing & Gallery Setup
        </Text>
        <Text style={styles.subtitle}>
          Start indexing your photos to load them all into the gallery
        </Text>
        <Button 
          mode="contained"
          onPress={() => startPhotoIndexing()}
          style={styles.button}
        >
          Start Indexing Photos
        </Button>
      </Card>
    );
  }

  return (
    <Card style={styles.card}>
      <Text style={styles.statusText}>
        {status}
      </Text>
      
      {totalFiles > 0 && (
        <Text style={styles.progressText}>
          {processedFiles} of {totalFiles} photos processed
        </Text>
      )}
      
      <ProgressBar 
        progress={progress / 100} 
        style={styles.progressBar}
        color="#007AFF"
      />
      
      <View style={styles.progressContainer}>
        <ActivityIndicator size="small" color="#007AFF" style={styles.spinner} />
        <Text style={styles.progressPercent}>
          {Math.round(progress)}% complete
        </Text>
      </View>
      
      <Button 
        mode="outlined"
        onPress={cancelIndexing}
        style={styles.cancelButton}
        buttonColor="#FF3B30"
        textColor="white"
      >
        Cancel
      </Button>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    margin: 16,
    padding: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  subtitle: {
    marginBottom: 16,
    textAlign: 'center',
    color: '#666',
  },
  button: {
    marginTop: 8,
  },
  statusText: {
    fontSize: 18,
    marginBottom: 16,
    textAlign: 'center',
    fontWeight: '500',
  },
  progressText: {
    marginBottom: 8,
    textAlign: 'center',
    color: '#666',
  },
  progressBar: {
    marginBottom: 16,
    height: 8,
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  spinner: {
    marginRight: 8,
  },
  progressPercent: {
    color: '#666',
  },
  cancelButton: {
    marginTop: 8,
  },
});
