import React from 'react';
import { View } from 'react-native';
import { Text, Button } from 'react-native-paper';

type Props = { title: string; description?: string; actionLabel?: string; onAction?: () => void };

export default function EmptyState({ title, description, actionLabel, onAction }: Props) {
  return (
    <View style={{ padding: 24, alignItems: 'center', justifyContent: 'center' }}>
      <Text variant="titleMedium" style={{ marginBottom: 8 }}>{title}</Text>
      {!!description && <Text style={{ marginBottom: 16, textAlign: 'center' }}>{description}</Text>}
      {!!actionLabel && <Button mode="contained" onPress={onAction}>{actionLabel}</Button>}
    </View>
  );
}






