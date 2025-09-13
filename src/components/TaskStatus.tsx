import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

type Props = { status?: { status?: string; progress?: number; total?: number; error?: string } | null };

export default function TaskStatus({ status }: Props) {
  if (!status) return null;
  const pct = typeof status.progress === 'number' ? (status.progress / 100) : 0;
  return (
    <View style={styles.container}>
      <Text style={styles.statusText}>Task: {status.status}</Text>
      {typeof status.progress === 'number' && (
        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, { width: `${pct * 100}%` }]} />
        </View>
      )}
      {!!status.error && <Text style={styles.errorText}>{status.error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 8,
  },
  statusText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  progressContainer: {
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginTop: 6,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  errorText: {
    color: 'red',
    fontSize: 14,
    marginTop: 4,
  },
});



