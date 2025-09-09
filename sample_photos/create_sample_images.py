#!/usr/bin/env python3
"""
Script to create 100 sample images with diverse content for Samsung PRISM MVP testing.
These images will have varied subjects to test the AI search functionality.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_sample_images():
    """Create 100 diverse sample images for testing."""
    
    # Define image categories and descriptions
    categories = {
        'nature': [
            'Sunset over mountains', 'Beach with waves', 'Forest with tall trees', 
            'Field of flowers', 'Lake reflection', 'Desert landscape', 'Waterfall',
            'Rainbow after rain', 'Snow-covered hills', 'Autumn leaves',
            'Cherry blossoms', 'Cactus in desert', 'Ocean waves', 'Pine forest',
            'Meadow with grass', 'River flowing', 'Cliff by sea', 'Garden flowers',
            'Mountain peak', 'Peaceful lake'
        ],
        'animals': [
            'Cat sitting', 'Dog playing', 'Bird flying', 'Fish swimming',
            'Horse running', 'Elephant walking', 'Tiger resting', 'Lion roaring',
            'Butterfly on flower', 'Rabbit hopping', 'Duck in pond', 'Owl perched',
            'Dolphin jumping', 'Bear in forest', 'Deer grazing', 'Penguin walking',
            'Monkey climbing', 'Snake slithering', 'Eagle soaring', 'Frog on lily pad'
        ],
        'food': [
            'Pizza slice', 'Burger and fries', 'Sushi platter', 'Fresh salad',
            'Chocolate cake', 'Ice cream cone', 'Coffee cup', 'Fruit bowl',
            'Pasta dish', 'Grilled chicken', 'Sandwich', 'Cookies',
            'Apple pie', 'Smoothie bowl', 'Bread loaf', 'Cheese platter',
            'Tacos', 'Ramen bowl', 'Pancakes', 'Wine glass'
        ],
        'urban': [
            'City skyline', 'Street with cars', 'Park bench', 'Building facade',
            'Traffic lights', 'Subway station', 'Bridge over river', 'Fountain',
            'Statue in square', 'Market stall', 'Cafe exterior', 'Bus stop',
            'Bicycle lane', 'Skyscraper', 'Shopping street', 'Church tower',
            'Museum entrance', 'Concert hall', 'Library steps', 'University campus'
        ],
        'people': [
            'Person reading', 'Child playing', 'Woman walking', 'Man cooking',
            'Friends talking', 'Family picnic', 'Athlete running', 'Artist painting',
            'Student studying', 'Worker building', 'Dancer performing', 'Singer on stage',
            'Teacher explaining', 'Doctor examining', 'Chef preparing', 'Photographer shooting',
            'Musician playing', 'Writer typing', 'Gardener planting', 'Driver steering'
        ]
    }
    
    # Color schemes for variety
    color_schemes = [
        ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        ['#A8E6CF', '#FFD93D', '#6BCF7F', '#4D96FF', '#9B59B6'],
        ['#FF8A80', '#82B1FF', '#B9F6CA', '#FFD54F', '#CE93D8'],
        ['#F48FB1', '#80CBC4', '#A5D6A7', '#FFCC02', '#B39DDB'],
        ['#FFAB91', '#81C784', '#64B5F6', '#FFB74D', '#F06292']
    ]
    
    # Create output directory
    output_dir = 'c:/Users/graph/Desktop/samsung_prism/SamsungPrism/assets/sample_images'
    os.makedirs(output_dir, exist_ok=True)
    
    image_count = 0
    
    for category, descriptions in categories.items():
        for i, description in enumerate(descriptions):
            if image_count >= 100:
                break
                
            # Create image
            width, height = 800, 600
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Select color scheme
            colors = random.choice(color_schemes)
            bg_color = colors[0]
            accent_color = colors[1]
            text_color = colors[2]
            
            # Draw background
            draw.rectangle([0, 0, width, height], fill=bg_color)
            
            # Draw some geometric shapes to make it visually interesting
            for _ in range(random.randint(3, 7)):
                shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
                color = random.choice(colors[1:])
                
                if shape_type == 'rectangle':
                    x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                    x2, y2 = random.randint(x1, width), random.randint(y1, height)
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=accent_color, width=2)
                elif shape_type == 'ellipse':
                    x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                    x2, y2 = random.randint(x1, width), random.randint(y1, height)
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline=accent_color, width=2)
            
            # Add category icon area
            icon_size = 80
            icon_x = width - icon_size - 20
            icon_y = 20
            draw.rectangle([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size], 
                         fill=accent_color, outline=text_color, width=3)
            
            # Add category text in icon
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw category text
            category_text = category.upper()
            bbox = draw.textbbox((0, 0), category_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = icon_x + (icon_size - text_width) // 2
            text_y = icon_y + (icon_size - text_height) // 2
            draw.text((text_x, text_y), category_text, fill='white', font=font)
            
            # Add main description text
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                desc_font = ImageFont.truetype("arial.ttf", 18)
            except:
                title_font = ImageFont.load_default()
                desc_font = ImageFont.load_default()
            
            # Main title
            title_bbox = draw.textbbox((0, 0), description, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            title_y = height // 2 - 40
            
            # Add text background
            padding = 20
            draw.rectangle([title_x - padding, title_y - padding, 
                          title_x + title_width + padding, title_y + 40 + padding],
                         fill='white', outline=text_color, width=2)
            
            draw.text((title_x, title_y), description, fill=text_color, font=title_font)
            
            # Add sample number
            sample_text = f"Sample #{image_count + 1:03d}"
            draw.text((20, height - 40), sample_text, fill=text_color, font=desc_font)
            
            # Save image
            filename = f"{image_count + 1:03d}_{category}_{description.lower().replace(' ', '_')}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, 'JPEG', quality=85)
            
            print(f"Created: {filename}")
            image_count += 1
            
            if image_count >= 100:
                break
    
    print(f"\nSuccessfully created {image_count} sample images!")
    
    # Create metadata file
    metadata_content = f"""# Samsung PRISM Sample Images

This directory contains {image_count} sample images for testing the AI-powered photo search functionality.

## Categories:
- Nature: Landscapes, weather, natural scenes
- Animals: Various wildlife and pets
- Food: Meals, snacks, beverages
- Urban: City scenes, architecture, infrastructure
- People: Human activities and interactions

## Usage:
These images are designed to test the conversational search capabilities of Samsung PRISM.
Try searches like:
- "show me nature photos"
- "find pictures of food"
- "photos with animals"
- "urban landscapes"
- "people doing activities"

Generated on: {os.path.basename(__file__)}
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(metadata_content)

if __name__ == "__main__":
    create_sample_images()
