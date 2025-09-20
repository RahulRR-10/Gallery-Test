import React, { useState } from 'react';
import { ScrollView, View, StyleSheet, Text, TextInput, TouchableOpacity, Alert } from 'react-native';
import { useMutation, useQuery } from '@tanstack/react-query';
import { getStatus, getStats, startIndexing, getTask } from '../services/api';
import TaskStatus from '../components/TaskStatus';
import FullscreenLoader from '../components/FullscreenLoader';
import AppHeader from '../components/AppHeader';

export default function SettingsScreen({ navigation }: any) {
  const [apiBase, setApiBase] = useState('http://localhost:8000');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<any>(null);

  const statusQ = useQuery({ 
    queryKey: ['status'], 
    queryFn: getStatus,
    onError: (error) => console.error('Status query error:', error),
    onSuccess: (data) => console.log('Status query success:', data)
  });
  const statsQ = useQuery({ 
    queryKey: ['stats'], 
    queryFn: getStats,
    onError: (error) => console.error('Stats query error:', error),
    onSuccess: (data) => console.log('Stats query success:', data)
  });

  const indexMutation = useMutation({
    mutationFn: ({ directory, recursive }: { directory: string; recursive: boolean }) => startIndexing(directory, recursive),
    onSuccess: (data) => {
      console.log('Indexing started successfully:', data);
      setTaskId(data.task_id);
    },
    onError: (error) => {
      console.error('Indexing error:', error);
    },
  });

  React.useEffect(() => {
    let t: any;
    if (taskId) {
      const poll = async () => {
        try {
          console.log('Polling task status for:', taskId);
          const s = await getTask(taskId);
          console.log('Task status:', s);
          setTaskStatus(s);
          if (s.status !== 'completed' && s.status !== 'failed') {
            t = setTimeout(poll, 1500);
          } else if (s.status === 'completed') {
            console.log('Task completed, refreshing status/stats');
            // Refresh status/stats when indexing finishes
            statusQ.refetch();
            statsQ.refetch();
          }
        } catch (error) {
          console.error('Task polling error:', error);
        }
      };
      poll();
    }
    return () => t && clearTimeout(t);
  }, [taskId]);

  return (
    <View style={styles.container}>
      <AppHeader title="Settings" />
      <FullscreenLoader 
        visible={!!taskId && taskStatus?.status === 'running'} 
        title="Indexing photos" 
        progress={taskStatus?.progress} 
      />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        {/* Backend Configuration */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Backend Configuration</Text>
          <TextInput 
            style={styles.input}
            value={apiBase} 
            onChangeText={setApiBase} 
            placeholder="http://192.168.1.10:8000"
            placeholderTextColor="#999"
          />
          <View style={styles.statusContainer}>
            {statusQ.isLoading ? (
              <Text style={styles.statusText}>Loading status...</Text>
            ) : (
              <Text style={styles.statusText}>
                Status: {statusQ.data?.status ?? '—'} | Photos: {statusQ.data?.photos_indexed ?? '—'}
              </Text>
            )}
            {statsQ.isLoading ? (
              <Text style={styles.statusText}>Loading stats...</Text>
            ) : (
              <Text style={styles.statusText}>
                Stats: photos {statsQ.data?.total_photos ?? '—'}, faces {statsQ.data?.total_faces ?? '—'}
              </Text>
            )}
            {statusQ.error && (
              <Text style={[styles.statusText, { color: 'red' }]}>
                Error: {statusQ.error.message}
              </Text>
            )}
            {statsQ.error && (
              <Text style={[styles.statusText, { color: 'red' }]}>
                Error: {statsQ.error.message}
              </Text>
            )}
          </View>
        </View>

        {/* Photo Indexing */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>Photo Indexing</Text>
          <TextInput 
            style={styles.input}
            placeholder="D:\\Photos or ./sample_photos" 
            placeholderTextColor="#999"
            onChangeText={() => {}} 
          />
          <TouchableOpacity 
            style={[styles.button, indexMutation.isLoading && styles.buttonDisabled]} 
            onPress={() => indexMutation.mutate({ directory: 'sample_photos', recursive: true })}
            disabled={indexMutation.isLoading}
          >
            <Text style={styles.buttonText}>
              {indexMutation.isLoading ? 'Starting...' : 'Start Indexing'}
            </Text>
          </TouchableOpacity>
          <TaskStatus status={taskStatus} />
        </View>

        {/* User Guide */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>User Guide</Text>
          <View style={styles.guideContainer}>
            <View style={styles.guideItem}>
              <Text style={styles.guideTitle}>1. Start API</Text>
              <Text style={styles.guideDescription}>
                Run python backend/start_api.py --host 0.0.0.0 --port 8000
              </Text>
            </View>
            <View style={styles.guideItem}>
              <Text style={styles.guideTitle}>2. Set App URL</Text>
              <Text style={styles.guideDescription}>
                Use your PC LAN IP (e.g. http://192.168.1.10:8000).
              </Text>
            </View>
            <View style={styles.guideItem}>
              <Text style={styles.guideTitle}>3. Search</Text>
              <Text style={styles.guideDescription}>
                Use the Search screen; add time filter if needed.
              </Text>
            </View>
            <View style={styles.guideItem}>
              <Text style={styles.guideTitle}>4. Gallery</Text>
              <Text style={styles.guideDescription}>
                Browse results grid; tap to view photo details.
              </Text>
            </View>
            <View style={styles.guideItem}>
              <Text style={styles.guideTitle}>5. Faces & Groups</Text>
              <Text style={styles.guideDescription}>
                Manage clusters and groups via backend endpoints.
              </Text>
            </View>
          </View>
        </View>

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 16,
  },
  card: {
    backgroundColor: 'white',
    marginBottom: 16,
    padding: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    fontSize: 16,
    backgroundColor: 'white',
    color: '#333',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  statusContainer: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  statusText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
    lineHeight: 20,
  },
  guideContainer: {
    marginTop: 8,
  },
  guideItem: {
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  guideTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  guideDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});

