import React, { useState } from 'react';
import { View } from 'react-native';
import { List, ActivityIndicator, Chip } from 'react-native-paper';
import { useMutation } from '@tanstack/react-query';
import { searchPhotos } from '../services/api';
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
  const [timeFilter, setTimeFilter] = useState('');
  const [person, setPerson] = useState('');
  const [group, setGroup] = useState('');
  const [relationship, setRelationship] = useState('');

  const mutation = useMutation({ mutationFn: (payload: any) => searchPhotos(payload) });

  const onSearch = () => mutation.mutate({
    query: query || null,
    person: person || null,
    group: group || null,
    relationship: relationship || null,
    time_filter: timeFilter || null,
    limit: 30,
  });

  const results = mutation.data?.results || [];

  return (
    <View style={{ flex: 1 }}>
      <AppHeader title="Photo Search" rightIcons={[{ name: 'cog', onPress: () => navigation.navigate('Settings') }]} />
      <SearchBar
        query={query}
        onQueryChange={setQuery}
        timeFilter={timeFilter}
        onTimeFilterChange={setTimeFilter}
        person={person}
        onPersonChange={setPerson}
        group={group}
        onGroupChange={setGroup}
        relationship={relationship}
        onRelationshipChange={setRelationship}
        onSearch={onSearch}
      />
      <List.Section>
        <List.Subheader>Quick filters</List.Subheader>
        <View style={{ paddingHorizontal: 12 }}>
          <Chip icon="calendar" mode="outlined" style={{ marginVertical: 6 }} onPress={() => setTimeFilter('today')}>Today</Chip>
          <Chip icon="calendar-week" mode="outlined" style={{ marginVertical: 6 }} onPress={() => setTimeFilter('last week')}>Last week</Chip>
          <Chip icon="calendar-month" mode="outlined" style={{ marginVertical: 6 }} onPress={() => setTimeFilter('last month')}>Last month</Chip>
        </View>
      </List.Section>
      {mutation.isLoading && <ActivityIndicator style={{ marginTop: 16 }} />}
      <List.Section>
        {results.map((p: any) => (
          <List.Item key={p.id} title={p.filename} description={p.path} onPress={() => navigation.navigate('PhotoViewer', { photoId: p.id, photo: p })} left={(props) => <List.Icon {...props} icon="image" />} />
        ))}
      </List.Section>
    </View>
  );
}

