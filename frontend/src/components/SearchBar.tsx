import React from 'react';
import { View } from 'react-native';
import { TextInput, IconButton, Text } from 'react-native-paper';

type Props = {
  query: string;
  onQueryChange: (v: string) => void;
  onSearch: () => void;
  isLoading?: boolean;
};

export default function SearchBar({ query, onQueryChange, onSearch, isLoading }: Props) {
  return (
    <View style={{ padding: 16, gap: 12 }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
        <TextInput
          mode="outlined"
          placeholder="Search everything... (e.g., 'John birthday cake 2024')"
          value={query}
          onChangeText={onQueryChange}
          style={{ flex: 1, borderRadius: 28 }}
          contentStyle={{ borderRadius: 28 }}
          outlineStyle={{ borderRadius: 28 }}
          right={<TextInput.Icon icon="magnify" onPress={onSearch} />}
          onSubmitEditing={onSearch}
          disabled={isLoading}
        />
        <IconButton 
          icon="send" 
          size={28} 
          onPress={onSearch}
          disabled={isLoading || !query.trim()}
          mode="contained"
        />
      </View>
      
      <Text variant="bodySmall" style={{ color: '#666', textAlign: 'center', marginTop: 4 }}>
        🧠 Smart search: Try "Alice beach 2024" or "birthday cake last month"
      </Text>
    </View>
  );
}


