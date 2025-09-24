import React, { useState } from 'react';
import { View } from 'react-native';
import { List, ActivityIndicator } from 'react-native-paper';
import { useMutation } from '@tanstack/react-query';
import { searchPhotos } from '../services/api';
import { getPhotoFilename, getPhotoDisplayPath } from '../utils/photoUtils';
import SearchBar from '../components/SearchBar';
import AppHeader from '../components/AppHeader';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

type RootStackParamList = {
  Root: undefined;
  PhotoViewer: { photoId: string; photo: any };
  Person: { cluster: any };
};

type RootTabParamList = {
  Gallery: { initialQuery?: string | null } | undefined;
  Search: undefined;
  People: undefined;
  Groups: undefined;
  Relationships: undefined;
  Settings: undefined;
};

type Props = NativeStackScreenProps<RootStackParamList, 'Root'> & {
  navigation: any; // Tab navigation
  route: any; // Tab route
};

export default function SearchScreen({ navigation }: Props) {
  const [query, setQuery] = useState('');

  const mutation = useMutation({
    mutationFn: (payload: any) => searchPhotos(payload),
    onSuccess: (data) => {
      console.log('🔍 Search Results:', {
        query: data.query,
        search_method: data.search_method,
        total_results: data.results?.length || 0,
        parsed_components: data.parsed_components || 'legacy',
        sample_results: data.results?.slice(0, 2) // First 2 results for debugging
      });
    },
    onError: (error) => {
      console.error('❌ Search failed:', error);
    },
  });

  const onSearch = () => {
    if (!query.trim()) {
      return;
    }

    const searchPayload = {
      query: query.trim(),
      limit: 30,
      similarity_threshold: 0.7,
    };
    
    console.log('🧠 Intelligent search payload:', searchPayload);
    mutation.mutate(searchPayload);
  };

  const results = mutation.data?.results || [];
  const searchMethod = mutation.data?.search_method || '';
  const message = mutation.data?.message || '';

  // Create a readable search method description
  const getSearchMethodDescription = (method: string) => {
    switch (method) {
      case 'person_only': return 'Person search';
      case 'person_object': return 'Person + Object search';
      case 'object_only': return 'Object search';
      case 'object_time': return 'Object + Time search';
      case 'semantic': return 'Semantic search';
      case 'auto_person_detection': return 'Auto-detected person';
      case 'recent_browse': return 'Recent photos';
      default: return method || 'Smart search';
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <AppHeader
        title="Photo Search"
        rightIcons={[
          { name: 'cog', onPress: () => navigation.navigate('Settings') },
        ]}
      />
      <SearchBar
        query={query}
        onQueryChange={setQuery}
        onSearch={onSearch}
        isLoading={mutation.isPending}
      />
      {mutation.isPending && <ActivityIndicator style={{ marginTop: 16 }} />}
      
      {mutation.error && (
        <List.Item
          title="Search Error"
          description={`Error: ${mutation.error.message}`}
          left={props => <List.Icon {...props} icon="alert" />}
        />
      )}
      
      {mutation.data && (
        <List.Subheader>
          Found {results.length} photos{' '}
          {mutation.data.search_method && `(${getSearchMethodDescription(mutation.data.search_method)})`}
          {mutation.data.message && results.length === 0 && ` - ${mutation.data.message}`}
        </List.Subheader>
      )}
      
      {mutation.data && results.length === 0 && mutation.data.message && (
        <List.Item
          title="No Photos Found"
          description={mutation.data.message}
          left={props => <List.Icon {...props} icon="information" />}
        />
      )}
      
      <List.Section>
        {results.map((p: any) => (
          <List.Item
            key={p.id}
            title={getPhotoFilename(p)}
            description={`${getPhotoDisplayPath(p)} • ${p.similarity_score ? (p.similarity_score * 100).toFixed(0) + '%' : 'Exact match'}`}
            onPress={() =>
              navigation.navigate('PhotoViewer', { photoId: p.id, photo: p })
            }
            left={props => <List.Icon {...props} icon="image" />}
          />
        ))}
      </List.Section>
    </View>
  );
}
