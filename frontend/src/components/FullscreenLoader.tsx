import React from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';

type Props = { visible: boolean; title?: string; progress?: number };

export default function FullscreenLoader({ visible, title, progress }: Props) {
  if (!visible) return null;
  return (
    <View style={styles.overlay}>
      <View style={styles.container}>
        {!!title && <Text style={styles.title}>{title}</Text>}
        <ActivityIndicator size="large" color="#007AFF" style={styles.spinner} />
        {typeof progress === 'number' && (
          <View style={styles.progressContainer}>
            <View style={[styles.progressBar, { width: `${progress}%` }]} />
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  container: {
    width: '85%',
    maxWidth: 360,
    backgroundColor: 'white',
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
    textAlign: 'center',
    color: '#333',
  },
  spinner: {
    marginBottom: 16,
  },
  progressContainer: {
    width: '100%',
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
});


