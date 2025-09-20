import React from 'react';
import { View } from 'react-native';
import { TextInput, IconButton } from 'react-native-paper';

type Props = {
  query: string;
  onQueryChange: (v: string) => void;
  timeFilter: string;
  onTimeFilterChange: (v: string) => void;
  person: string;
  onPersonChange: (v: string) => void;
  group: string;
  onGroupChange: (v: string) => void;
  relationship: string;
  onRelationshipChange: (v: string) => void;
  onSearch: () => void;
};

export default function SearchBar({ query, onQueryChange, timeFilter, onTimeFilterChange, person, onPersonChange, group, onGroupChange, relationship, onRelationshipChange, onSearch }: Props) {
  return (
    <View style={{ padding: 12, gap: 12 }}>
      <TextInput
        mode="outlined"
        placeholder="Search photos (e.g. sunset beach)"
        value={query}
        onChangeText={onQueryChange}
        right={<TextInput.Icon icon="magnify" onPress={onSearch} />}
        style={{ borderRadius: 24 }}
        contentStyle={{ borderRadius: 24 }}
        outlineStyle={{ borderRadius: 24 }}
      />
      <View style={{ flexDirection: 'row', gap: 8 }}>
        <TextInput
          mode="outlined"
          placeholder="Time filter (e.g. last month)"
          value={timeFilter}
          onChangeText={onTimeFilterChange}
          style={{ flex: 1, borderRadius: 24 }}
          contentStyle={{ borderRadius: 24 }}
          outlineStyle={{ borderRadius: 24 }}
        />
        <IconButton icon="arrow-right-circle" size={28} onPress={onSearch} />
      </View>
      <TextInput
        mode="outlined"
        placeholder="Person (e.g. Alice)"
        value={person}
        onChangeText={onPersonChange}
        style={{ borderRadius: 24 }}
        contentStyle={{ borderRadius: 24 }}
        outlineStyle={{ borderRadius: 24 }}
      />
      <View style={{ flexDirection: 'row', gap: 8 }}>
        <TextInput
          mode="outlined"
          placeholder="Group (e.g. family)"
          value={group}
          onChangeText={onGroupChange}
          style={{ flex: 1, borderRadius: 24 }}
          contentStyle={{ borderRadius: 24 }}
          outlineStyle={{ borderRadius: 24 }}
        />
        <TextInput
          mode="outlined"
          placeholder="Relationship (e.g. friends)"
          value={relationship}
          onChangeText={onRelationshipChange}
          style={{ flex: 1, borderRadius: 24 }}
          contentStyle={{ borderRadius: 24 }}
          outlineStyle={{ borderRadius: 24 }}
        />
      </View>
    </View>
  );
}


