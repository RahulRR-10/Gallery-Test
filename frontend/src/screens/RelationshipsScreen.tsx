import React from 'react';
import { View } from 'react-native';
import { ActivityIndicator, List, Button } from 'react-native-paper';
import { useQuery } from '@tanstack/react-query';
import { getRelationships, buildRelationships, getTask } from '../services/api';
import { useState } from 'react';
import AppHeader from '../components/AppHeader';
import FullscreenLoader from '../components/FullscreenLoader';

export default function RelationshipsScreen({ navigation }: any) {
  const { data, isLoading, refetch } = useQuery({ queryKey: ['relationships'], queryFn: getRelationships });
  const relationships = data?.relationships || [];
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<any>(null);

  const onBuild = async () => {
    try {
      const res = await buildRelationships();
      setTaskId(res.task_id);
    } catch {}
  };

  React.useEffect(() => {
    let t: any;
    if (taskId) {
      const poll = async () => {
        try {
          const s = await getTask(taskId);
          setTaskStatus(s);
          if (s.status !== 'completed' && s.status !== 'failed') {
            t = setTimeout(poll, 1500);
          } else if (s.status === 'completed') {
            refetch();
          }
        } catch {}
      };
      poll();
    }
    return () => t && clearTimeout(t);
  }, [taskId]);

  return (
    <View style={{ flex: 1 }}>
      <AppHeader title="Relationships" />
      <FullscreenLoader visible={!!taskId && taskStatus?.status === 'running'} title="Building relationships" progress={taskStatus?.progress} />
      {isLoading && <ActivityIndicator style={{ marginTop: 16 }} />}
      <List.Section>
        {relationships.map((r: any, idx: number) => (
          <List.Item key={idx} title={`${r.person1 ?? ''} - ${r.person2 ?? ''}`} description={`${r.type ?? ''} (${r.confidence ?? ''})`} left={(props) => <List.Icon {...props} icon="link" />} />
        ))}
      </List.Section>
      <Button style={{ margin: 12 }} mode="contained" onPress={onBuild}>Build Relationships</Button>
    </View>
  );
}

