import React from 'react';
import { View } from 'react-native';
import { ActivityIndicator, List, TextInput, Button } from 'react-native-paper';
import { useQuery } from '@tanstack/react-query';
import { getGroups, createGroup } from '../services/api';
import { useState } from 'react';
import AppHeader from '../components/AppHeader';

export default function GroupsScreen({ navigation }: any) {
  const { data, isLoading } = useQuery({ queryKey: ['groups'], queryFn: getGroups });
  const groups = data?.groups || [];
  const [groupName, setGroupName] = useState('');
  const [members, setMembers] = useState('');
  const onCreate = async () => {
    const ids = members.split(',').map(s => s.trim()).filter(Boolean);
    if (!groupName || ids.length === 0) return;
    try { await createGroup(groupName, ids); } catch {}
  };

  return (
    <View style={{ flex: 1 }}>
      <AppHeader title="Groups" />
      {isLoading && <ActivityIndicator style={{ marginTop: 16 }} />}
      <List.Section>
        {groups.map((g: any, idx: number) => (
          <List.Item key={idx} title={g.name || g.group_name || 'Group'} description={`Members: ${g.members?.length ?? ''}`} left={(props) => <List.Icon {...props} icon="account-group" />} />
        ))}
      </List.Section>
      <List.Section>
        <TextInput mode="outlined" label="New group name" value={groupName} onChangeText={setGroupName} style={{ marginHorizontal: 12 }} />
        <TextInput mode="outlined" label="Member cluster IDs (comma separated)" value={members} onChangeText={setMembers} style={{ margin: 12 }} />
        <Button style={{ marginHorizontal: 12 }} mode="contained" onPress={onCreate}>Create Group</Button>
      </List.Section>
    </View>
  );
}

