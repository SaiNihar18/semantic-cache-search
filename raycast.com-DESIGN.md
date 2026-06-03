# DESIGN SYSTEM: Premium Semantic Cache Dashboard

## Overview
This design system optimizes for a high-end, futuristic, and premium aesthetic. It transitions from a flat navy/red theme to a stunning layered interface featuring deep space backgrounds, vibrant teal and indigo gradients, heavy glassmorphism, subtle glowing borders, and precise typography. This is designed to create an immediate "wow" factor.

## Colors

### Core Backgrounds & Surfaces
- `color-bg-app`: `#0A0A10` (Deepest void black/navy)
- `color-surface-base`: `rgba(25, 25, 35, 0.4)` (Translucent glass surface)
- `color-surface-highlight`: `rgba(40, 40, 55, 0.6)` (Hover state for glass)
- `color-border-glass`: `rgba(255, 255, 255, 0.08)` (Must be applied to all cards/inputs)
- `color-border-glow`: `rgba(0, 229, 255, 0.3)` (Active input focus)

### Premium Gradients & Accents
- `gradient-primary`: `linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%)` (Vibrant cyan/teal to blue)
- `gradient-secondary`: `linear-gradient(135deg, #667EEA 0%, #764BA2 100%)` (Indigo to deep purple)
- `color-accent-teal`: `#00F2FE`

### Typography Colors
- `color-text-primary`: `#FFFFFF`
- `color-text-secondary`: `#8B949E` (Cool muted gray, highly legible)
- `color-text-placeholder`: `#484F58`

### Semantic Feedback
- `color-hit-bg`: `rgba(46, 160, 67, 0.15)`
- `color-hit-border`: `rgba(46, 160, 67, 0.4)`
- `color-hit-text`: `#56D364`
- `color-miss-bg`: `rgba(248, 81, 73, 0.15)`
- `color-miss-border`: `rgba(248, 81, 73, 0.4)`
- `color-miss-text`: `#FF7B72`

## Typography
Family: 'Inter', 'SF Pro Display', sans-serif
- **Hero Title**: 32px, 700 weight, -2% tracking
- **Card Title**: 16px, 600 weight, 1.2 line height
- **Body / Input**: 14px, 400 weight, 1.5 line height
- **Badge / Detail**: 12px, 500 weight, 1px letter-spacing (uppercase)

## Visual Effects (CRITICAL FOR PREMIUM LOOK)
1. **Glassmorphism**: ALL cards, floating panels, and input fields MUST use `backdrop-filter: blur(24px)`. They must never be flat opaque colors.
2. **Inner Borders**: Every glass component must have a 1px solid `color-border-glass` to simulate light catching the edge of the glass.
3. **Subtle Shadows**: Use `box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4)` for floating elements.
4. **Gradient Text**: Large headings can optionally clip `gradient-primary` to the text.

## Border Radius
- `radius-sm`: 8px (Pill tags, badges, buttons)
- `radius-md`: 12px (Inputs, list items)
- `radius-lg`: 24px (Main container cards, modals)

## Spacing & Density
- Use an 8px baseline grid (8, 16, 24, 32, 40).
- Keep information dense but give containers breathing room (e.g., 24px padding inside main cards).