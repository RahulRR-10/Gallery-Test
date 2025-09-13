import React from 'react';
import { Appbar, useTheme } from 'react-native-paper';

type Props = {
  title: string;
  onBack?: () => void;
  rightIcons?: Array<{ name: string; onPress: () => void }>;
};

export default function AppHeader({ title, onBack, rightIcons }: Props) {
  const theme = useTheme();
  const bg = theme.colors.primary;
  const fg = theme.colors.onPrimary;
  return (
    <Appbar.Header style={{ backgroundColor: bg }}>
      {onBack ? <Appbar.BackAction color={fg} onPress={onBack} /> : null}
      <Appbar.Content color={fg} title={title} />
      {rightIcons?.map((i, idx) => (
        <Appbar.Action key={idx} color={fg} icon={i.name} onPress={i.onPress} />
      ))}
    </Appbar.Header>
  );
}

