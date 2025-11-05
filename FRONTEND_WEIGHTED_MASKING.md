# Frontend Weighted Masking Implementation

## Changes Made

### 1. Training Form (`frontend/src/routes/training/new/+page.svelte`)

**Added Form State:**
```typescript
selective_loss_structural_weight: 0.1,  // NEW field
```

**Updated Strategy Selector:**
- Added "weighted" option to strategy dropdown
- Three choices now available: epoch_based, alternating, weighted

**Conditional Controls:**
The form now shows different controls based on the selected strategy:

#### Epoch-Based Strategy
- Shows: Masking Start Epoch slider (0.0 to num_epochs)
- Color: Green accent

#### Alternating Strategy  
- Shows: Cycle Length slider (20-500 steps)
- Shows: Masking ON Duration slider (10 to cycle_length)
- Shows: Pattern visualization with percentages
- Color: Blue/Purple accents

#### Weighted Strategy (NEW)
- Shows: Structural Token Weight slider (0.0 to 1.0, step 0.05)
- Shows: Current weighting visualization
- Shows: Contextual recommendations based on weight value
- Color: Amber accent

**Weight Value Recommendations:**
- < 0.1: "Very low structure emphasis - model may struggle with formatting"
- 0.1-0.3: "Recommended for structured outputs - good balance"
- 0.3-0.7: "Moderate structure emphasis - more balanced training"
- > 0.7: "High structure emphasis - close to unweighted training"

**Visual Indicators:**
- 🔧 Structural tokens weight visualization
- 📝 Semantic tokens (always 1.0×)
- 💡 Tips for choosing weight values

### 2. Form Submission

Updated the API request payload to include:
```typescript
selective_loss_structural_weight: formData.selective_loss_structural_weight
```

## UI/UX Features

### Strategy Descriptions
Each strategy now has an emoji and clear description:
- 📅 **Epoch-based**: Enable masking after a certain epoch
- 🔄 **Alternating**: Continuously cycle between learning structure and semantics
- ⚖️ **Weighted**: Soft masking with reduced weight for structural tokens

### Interactive Controls
- Range sliders with clear min/max labels
- Real-time value display
- Color-coded accents matching strategy type
- Contextual help text and recommendations

### Responsive Info Boxes
- Color-coded backgrounds (green/purple/amber) matching strategy
- Dynamic content based on current settings
- Tips and best practices

## Testing

### Build Status
✅ Frontend builds successfully with no errors

### Files Modified
- `frontend/src/routes/training/new/+page.svelte`

### Build Output
- Total build time: ~3.87s
- Largest bundle: `training/new/_page.svelte.js` (22.35 kB)
- No build warnings or errors

## User Flow

1. User enables "Selective Loss" checkbox
2. User selects masking strategy from dropdown
3. **Only relevant controls for that strategy appear**
4. User adjusts strategy-specific parameters with visual feedback
5. Form submits with all parameters to API

## Default Values

- **selective_loss_masking_strategy**: `"epoch_based"`
- **selective_loss_structural_weight**: `0.1` (10% weight for structural tokens)

## API Integration

The form now sends the complete weighted masking configuration:
```json
{
  "selective_loss": true,
  "selective_loss_masking_strategy": "weighted",
  "selective_loss_structural_weight": 0.1,
  // ... other fields
}
```

## Benefits

1. **Clean UI**: Only shows controls relevant to selected strategy
2. **No Confusion**: Users don't see epoch controls when using weighted strategy
3. **Visual Feedback**: Real-time weight visualization helps users understand impact
4. **Best Practices**: Built-in recommendations guide users to good defaults
5. **Progressive Disclosure**: Advanced controls only appear when needed

## Screenshot Descriptions

### Epoch-Based Strategy
- Green-themed controls
- Single slider for start epoch
- Shows when masking begins

### Alternating Strategy
- Blue/Purple-themed controls
- Two sliders: cycle length and masking duration
- Pattern visualization showing percentage split

### Weighted Strategy (NEW)
- Amber-themed controls
- Single slider for structural weight (0.0-1.0)
- Weight distribution visualization
- Contextual recommendations
- Tips box with best practices

## Status: ✅ Complete

Frontend fully implements weighted masking strategy with conditional controls!
