/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import { StatusBar } from 'react-native';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider, MD3LightTheme } from 'react-native-paper';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SearchScreen from './src/screens/SearchScreen';
import GalleryScreen from './src/screens/GalleryScreen';
import PhotoViewer from './src/screens/PhotoViewer';
import SettingsScreen from './src/screens/SettingsScreen';
import PeopleScreen from './src/screens/PeopleScreen';
import PersonScreen from './src/screens/PersonScreen';
import GroupsScreen from './src/screens/GroupsScreen';
import RelationshipsScreen from './src/screens/RelationshipsScreen';

const Stack = createNativeStackNavigator();
const Tabs = createBottomTabNavigator();
const queryClient = new QueryClient();

export default function App() {
  return (
    <PaperProvider theme={{
      ...MD3LightTheme,
      roundness: 20,
      colors: {
        ...MD3LightTheme.colors,
        primary: '#2563EB',
        secondary: '#0EA5E9',
        surfaceVariant: '#F1F5F9',
        outline: '#E2E8F0',
        onPrimary: '#FFFFFF',
      },
    }}>
      <QueryClientProvider client={queryClient}>
        <NavigationContainer theme={{ ...DefaultTheme, colors: { ...DefaultTheme.colors, background: '#F8FAFC' } }}>
          <StatusBar barStyle="dark-content" />
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Root" component={RootTabs} />
            <Stack.Screen name="PhotoViewer" component={PhotoViewer} options={{ title: 'Photo' }} />
            <Stack.Screen name="Person" component={PersonScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </QueryClientProvider>
    </PaperProvider>
  );
}

function RootTabs() {
  return (
    <Tabs.Navigator screenOptions={{ headerShown: false, tabBarActiveTintColor: '#4F46E5' }}>
      <Tabs.Screen name="Gallery" component={GalleryScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="image-multiple" color={color} size={size} />) }} />
      <Tabs.Screen name="Search" component={SearchScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="magnify" color={color} size={size} />) }} />
      <Tabs.Screen name="People" component={PeopleScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="account" color={color} size={size} />) }} />
      <Tabs.Screen name="Groups" component={GroupsScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="account-group" color={color} size={size} />) }} />
      <Tabs.Screen name="Relationships" component={RelationshipsScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="link" color={color} size={size} />) }} />
      <Tabs.Screen name="Settings" component={SettingsScreen} options={{ tabBarIcon: ({ color, size }) => (<MaterialCommunityIcons name="cog" color={color} size={size} />) }} />
    </Tabs.Navigator>
  );
}
