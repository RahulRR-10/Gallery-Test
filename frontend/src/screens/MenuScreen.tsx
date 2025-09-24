import React from 'react';
import { View, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { Text, Surface, useTheme } from 'react-native-paper';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { useNavigation } from '@react-navigation/native';

const MenuScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();

  const menuItems = [
    {
      title: 'Videos',
      icon: 'play-circle-outline',
      onPress: () => {},
    },
    {
      title: 'Favourites', 
      icon: 'heart-outline',
      onPress: () => {},
    },
    {
      title: 'Recent',
      icon: 'clock-outline', 
      onPress: () => {},
    },
    {
      title: 'Clean out',
      icon: 'delete-outline',
      onPress: () => {},
    },
    {
      title: 'Locations',
      icon: 'map-marker-outline',
      onPress: () => {},
    },
    {
      title: 'Shared albums',
      icon: 'account-group-outline',
      onPress: () => {},
    },
    {
      title: 'Recycle bin',
      icon: 'delete-restore',
      onPress: () => {},
    },
    {
      title: 'Settings',
      icon: 'cog-outline',
      onPress: () => navigation.navigate('Settings'),
    },
  ];

  const aiFeatures = [
    {
      title: 'Groups',
      icon: 'account-group-outline',
      subtitle: 'Photo groupings and collections', 
      onPress: () => navigation.navigate('Groups'),
    },
    {
      title: 'Relationships',
      icon: 'link-variant',
      subtitle: 'Person relationships and connections',
      onPress: () => navigation.navigate('Relationships'),
    },
  ];

  const renderMenuItem = (item, index, showSubtitle = false) => (
    <TouchableOpacity
      key={index}
      style={[styles.menuItem, { borderBottomColor: theme.colors.outline }]}
      onPress={item.onPress}
    >
      <View style={styles.menuItemContent}>
        <MaterialCommunityIcons 
          name={item.icon} 
          size={24} 
          color={theme.colors.onSurface}
          style={styles.menuIcon}
        />
        <View style={styles.menuTextContainer}>
          <Text style={[styles.menuItemText, { color: theme.colors.onSurface }]}>
            {item.title}
          </Text>
          {showSubtitle && item.subtitle && (
            <Text style={[styles.menuItemSubtitle, { color: theme.colors.onSurfaceVariant }]}>
              {item.subtitle}
            </Text>
          )}
        </View>
      </View>
      <MaterialCommunityIcons 
        name="chevron-right" 
        size={20} 
        color={theme.colors.onSurfaceVariant}
      />
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Standard Menu Items */}
        <Surface style={[styles.section, { backgroundColor: theme.colors.surface }]}>
          {menuItems.map((item, index) => renderMenuItem(item, index))}
        </Surface>

        {/* AI Features Section */}
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.colors.onBackground }]}>
            AI Features
          </Text>
        </View>
        <Surface style={[styles.section, { backgroundColor: theme.colors.surface }]}>
          {aiFeatures.map((item, index) => renderMenuItem(item, `ai-${index}`, true))}
        </Surface>

        {/* Go to Studio Section */}
        <TouchableOpacity style={[styles.studioButton, { backgroundColor: theme.colors.surface }]}>
          <View style={styles.studioContent}>
            <View style={[styles.studioIcon, styles.studioIconColor]}>
              <MaterialCommunityIcons name="camera" size={20} color="#FFFFFF" />
            </View>
            <Text style={[styles.studioText, { color: theme.colors.onSurface }]}>
              Go to Studio
            </Text>
          </View>
          <MaterialCommunityIcons 
            name="chevron-right" 
            size={20} 
            color={theme.colors.onSurfaceVariant}
          />
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    paddingVertical: 16,
  },
  section: {
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  sectionHeader: {
    marginHorizontal: 32,
    marginBottom: 8,
    marginTop: 16,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '500',
    opacity: 0.7,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 0.5,
  },
  menuItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  menuIcon: {
    marginRight: 16,
  },
  menuTextContainer: {
    flex: 1,
  },
  menuItemText: {
    fontSize: 16,
    fontWeight: '400',
  },
  menuItemSubtitle: {
    fontSize: 12,
    marginTop: 2,
    opacity: 0.7,
  },
  studioButton: {
    marginHorizontal: 16,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 16,
  },
  studioContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  studioIcon: {
    width: 32,
    height: 32,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  studioIconColor: {
    backgroundColor: '#FF4081',
  },
  studioText: {
    fontSize: 16,
    fontWeight: '500',
  },
});

export default MenuScreen;