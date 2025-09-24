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
import PicturesScreen from './src/screens/PicturesScreen';
import AlbumsScreen from './src/screens/AlbumsScreen';
import MenuScreen from './src/screens/MenuScreen';
import PhotoViewer from './src/screens/PhotoViewer';
import SettingsScreen from './src/screens/SettingsScreen';
import PeopleScreen from './src/screens/PeopleScreen';
import PersonScreen from './src/screens/PersonScreen';
import GroupsScreen from './src/screens/GroupsScreen';
import RelationshipsScreen from './src/screens/RelationshipsScreen';

const Stack = createNativeStackNavigator();
const Tabs = createBottomTabNavigator();
const queryClient = new QueryClient();

// Samsung Gallery Dark Theme
const SamsungTheme = {
  ...MD3LightTheme,
  dark: true,
  roundness: 8,
  colors: {
    ...MD3LightTheme.colors,
    primary: '#4285F4',
    secondary: '#34A853',
    background: '#000000',
    surface: '#1A1A1A',
    surfaceVariant: '#2D2D2D',
    outline: '#424242',
    onPrimary: '#FFFFFF',
    onBackground: '#FFFFFF',
    onSurface: '#FFFFFF',
    text: '#FFFFFF',
  },
};

const SamsungNavigationTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    background: '#000000',
    card: '#1A1A1A',
    text: '#FFFFFF',
    border: '#424242',
    notification: '#4285F4',
  },
};

export default function App() {
  return (
    <PaperProvider theme={SamsungTheme}>
      <QueryClientProvider client={queryClient}>
        <NavigationContainer theme={SamsungNavigationTheme}>
          <StatusBar barStyle="light-content" backgroundColor="#000000" />
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Root" component={RootTabs} />
            <Stack.Screen 
              name="PhotoViewer" 
              component={PhotoViewer} 
              options={{ 
                title: 'Photo',
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }} 
            />
            <Stack.Screen 
              name="Person" 
              component={PersonScreen}
              options={{
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }}
            />
            <Stack.Screen 
              name="People" 
              component={PeopleScreen}
              options={{
                title: 'People',
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }}
            />
            <Stack.Screen 
              name="Groups" 
              component={GroupsScreen}
              options={{
                title: 'Groups',
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }}
            />
            <Stack.Screen 
              name="Relationships" 
              component={RelationshipsScreen}
              options={{
                title: 'Relationships',
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }}
            />
            <Stack.Screen 
              name="Settings" 
              component={SettingsScreen}
              options={{
                title: 'Settings',
                headerShown: true,
                headerStyle: { backgroundColor: '#1A1A1A' },
                headerTintColor: '#FFFFFF',
              }}
            />
          </Stack.Navigator>
        </NavigationContainer>
      </QueryClientProvider>
    </PaperProvider>
  );
}

// Tab Icons Components (to fix the React component warning)
const PicturesIcon = ({ color, size }: { color: string; size: number }) => (
  <MaterialCommunityIcons name="image-multiple" color={color} size={size} />
);

const AlbumsIcon = ({ color, size }: { color: string; size: number }) => (
  <MaterialCommunityIcons name="folder-multiple-outline" color={color} size={size} />
);

const PeopleIcon = ({ color, size }: { color: string; size: number }) => (
  <MaterialCommunityIcons name="account-multiple" color={color} size={size} />
);

const MenuIcon = ({ color, size }: { color: string; size: number }) => (
  <MaterialCommunityIcons name="menu" color={color} size={size} />
);

function RootTabs() {
  return (
    <Tabs.Navigator 
      screenOptions={{ 
        headerShown: false, 
        tabBarActiveTintColor: '#4285F4',
        tabBarInactiveTintColor: '#888888',
        tabBarStyle: {
          backgroundColor: '#1A1A1A',
          borderTopColor: '#424242',
          borderTopWidth: 0.5,
          paddingBottom: 8,
          paddingTop: 8,
          height: 60,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '500',
        },
      }}
      initialRouteName="Pictures"
    >
      <Tabs.Screen 
        name="Pictures" 
        component={PicturesScreen} 
        options={{ 
          tabBarIcon: PicturesIcon,
          title: 'Pictures',
        }} 
      />
      <Tabs.Screen 
        name="Albums" 
        component={AlbumsScreen} 
        options={{ 
          tabBarIcon: AlbumsIcon,
          title: 'Albums',
        }} 
      />
      <Tabs.Screen 
        name="People" 
        component={PeopleScreen} 
        options={{ 
          tabBarIcon: PeopleIcon,
          title: 'People',
        }} 
      />
      <Tabs.Screen 
        name="Menu" 
        component={MenuScreen} 
        options={{ 
          tabBarIcon: MenuIcon,
          title: 'Menu',
        }} 
      />
    </Tabs.Navigator>
  );
}
