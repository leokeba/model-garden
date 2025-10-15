# Frontend Design Guidelines

## Overview
This document defines the frontend architecture, UI/UX design principles, component structure, and user workflows for Model Garden's web interface.

---

## 1. Technology Stack

### Core Technologies
- **Framework**: SvelteKit 2.0
- **Styling**: TailwindCSS 3.x
- **Type Safety**: TypeScript
- **Build Tool**: Vite
- **State Management**: Svelte stores
- **HTTP Client**: Fetch API / Axios
- **WebSocket**: Native WebSocket API
- **Charts**: Chart.js or D3.js
- **Icons**: Lucide Svelte or Heroicons

### Why These Choices?

#### SvelteKit
- Minimal bundle size
- True reactivity without virtual DOM
- Built-in SSR and routing
- Excellent developer experience
- Fast compilation

#### TailwindCSS
- Utility-first approach
- Rapid development
- Consistent design system
- Small production bundle
- Easy customization

---

## 2. Design System

### Color Palette

```css
/* Primary - Blue/Purple gradient */
--color-primary-50: #f0f9ff;
--color-primary-100: #e0f2fe;
--color-primary-500: #3b82f6;  /* Main primary */
--color-primary-600: #2563eb;
--color-primary-700: #1d4ed8;

/* Success - Green */
--color-success-500: #10b981;
--color-success-600: #059669;

/* Warning - Orange */
--color-warning-500: #f59e0b;
--color-warning-600: #d97706;

/* Error - Red */
--color-error-500: #ef4444;
--color-error-600: #dc2626;

/* Neutral - Gray */
--color-gray-50: #f9fafb;
--color-gray-100: #f3f4f6;
--color-gray-500: #6b7280;
--color-gray-700: #374151;
--color-gray-900: #111827;

/* Carbon - Emerald (for sustainability theme) */
--color-carbon-500: #34d399;
--color-carbon-600: #10b981;
```

### Typography

```css
/* Font Family */
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;

/* Font Sizes */
--text-xs: 0.75rem;    /* 12px */
--text-sm: 0.875rem;   /* 14px */
--text-base: 1rem;     /* 16px */
--text-lg: 1.125rem;   /* 18px */
--text-xl: 1.25rem;    /* 20px */
--text-2xl: 1.5rem;    /* 24px */
--text-3xl: 1.875rem;  /* 30px */
--text-4xl: 2.25rem;   /* 36px */

/* Font Weights */
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

### Spacing
Following Tailwind's spacing scale (4px base):
- `1` = 0.25rem (4px)
- `2` = 0.5rem (8px)
- `4` = 1rem (16px)
- `6` = 1.5rem (24px)
- `8` = 2rem (32px)

### Shadows
```css
--shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
--shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
```

### Border Radius
```css
--radius-sm: 0.25rem;   /* 4px */
--radius: 0.375rem;     /* 6px */
--radius-md: 0.5rem;    /* 8px */
--radius-lg: 0.75rem;   /* 12px */
--radius-xl: 1rem;      /* 16px */
```

---

## 3. Component Library

### Base Components

#### Button
```svelte
<!-- src/lib/components/Button.svelte -->
<script lang="ts">
  export let variant: 'primary' | 'secondary' | 'danger' | 'ghost' = 'primary';
  export let size: 'sm' | 'md' | 'lg' = 'md';
  export let disabled = false;
  export let loading = false;
  export let fullWidth = false;
</script>

<button
  class="btn btn-{variant} btn-{size}"
  class:w-full={fullWidth}
  {disabled}
  on:click
>
  {#if loading}
    <Spinner size="sm" />
  {/if}
  <slot />
</button>

<style lang="postcss">
  .btn {
    @apply inline-flex items-center justify-center gap-2 rounded-lg font-medium
           transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2;
  }
  
  .btn-primary {
    @apply bg-primary-600 text-white hover:bg-primary-700 
           focus:ring-primary-500;
  }
  
  .btn-sm {
    @apply px-3 py-1.5 text-sm;
  }
  
  .btn-md {
    @apply px-4 py-2 text-base;
  }
  
  .btn:disabled {
    @apply opacity-50 cursor-not-allowed;
  }
</style>
```

#### Card
```svelte
<!-- src/lib/components/Card.svelte -->
<script lang="ts">
  export let padding = true;
  export let hoverable = false;
</script>

<div 
  class="card"
  class:p-6={padding}
  class:hoverable
>
  <slot />
</div>

<style lang="postcss">
  .card {
    @apply bg-white rounded-lg shadow border border-gray-200;
  }
  
  .hoverable {
    @apply transition-shadow hover:shadow-md cursor-pointer;
  }
</style>
```

#### Badge
```svelte
<!-- src/lib/components/Badge.svelte -->
<script lang="ts">
  export let variant: 'success' | 'warning' | 'error' | 'info' = 'info';
  export let size: 'sm' | 'md' = 'md';
</script>

<span class="badge badge-{variant} badge-{size}">
  <slot />
</span>

<style lang="postcss">
  .badge {
    @apply inline-flex items-center gap-1 rounded-full font-medium;
  }
  
  .badge-sm {
    @apply px-2 py-0.5 text-xs;
  }
  
  .badge-md {
    @apply px-3 py-1 text-sm;
  }
  
  .badge-success {
    @apply bg-success-100 text-success-700;
  }
  
  .badge-error {
    @apply bg-error-100 text-error-700;
  }
</style>
```

### Complex Components

#### ModelCard
```svelte
<!-- src/lib/components/ModelCard.svelte -->
<script lang="ts">
  import type { Model } from '$lib/types';
  import { formatBytes, formatDate } from '$lib/utils';
  
  export let model: Model;
  export let onDeploy: () => void;
  export let onDelete: () => void;
</script>

<Card hoverable>
  <div class="space-y-4">
    <!-- Header -->
    <div class="flex items-start justify-between">
      <div>
        <h3 class="text-lg font-semibold text-gray-900">
          {model.name}
        </h3>
        <p class="text-sm text-gray-500">{model.base_model}</p>
      </div>
      <Badge variant={model.status === 'available' ? 'success' : 'warning'}>
        {model.status}
      </Badge>
    </div>
    
    <!-- Metrics -->
    <div class="grid grid-cols-2 gap-4">
      <div>
        <div class="text-xs text-gray-500">Size</div>
        <div class="font-medium">{formatBytes(model.size_bytes)}</div>
      </div>
      <div>
        <div class="text-xs text-gray-500">Carbon</div>
        <div class="font-medium text-carbon-600">
          {model.carbon_emissions.training_kg.toFixed(3)} kg CO₂
        </div>
      </div>
    </div>
    
    <!-- Actions -->
    <div class="flex gap-2">
      <Button variant="primary" size="sm" on:click={onDeploy}>
        Deploy
      </Button>
      <Button variant="ghost" size="sm">
        Details
      </Button>
      <Button variant="danger" size="sm" on:click={onDelete}>
        Delete
      </Button>
    </div>
  </div>
</Card>
```

#### TrainingStatus
```svelte
<!-- src/lib/components/TrainingStatus.svelte -->
<script lang="ts">
  import type { TrainingJob } from '$lib/types';
  import { ProgressBar, MetricsDisplay } from '$lib/components';
  
  export let job: TrainingJob;
</script>

<Card>
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <h3 class="text-lg font-semibold">{job.name}</h3>
      <Badge variant={job.status === 'completed' ? 'success' : 'info'}>
        {job.status}
      </Badge>
    </div>
    
    <ProgressBar 
      value={job.progress * 100} 
      max={100}
      label="Progress: {Math.round(job.progress * 100)}%"
    />
    
    <div class="grid grid-cols-3 gap-4 text-sm">
      <div>
        <div class="text-gray-500">Epoch</div>
        <div class="font-medium">{job.current_epoch}/{job.total_epochs}</div>
      </div>
      <div>
        <div class="text-gray-500">Steps</div>
        <div class="font-medium">{job.current_step}/{job.total_steps}</div>
      </div>
      <div>
        <div class="text-gray-500">Loss</div>
        <div class="font-medium">{job.metrics?.train_loss?.toFixed(4) ?? '-'}</div>
      </div>
    </div>
    
    {#if job.carbon}
      <div class="pt-4 border-t">
        <div class="flex items-center gap-2 text-sm">
          <span class="text-gray-500">Carbon Emissions:</span>
          <span class="font-semibold text-carbon-600">
            {job.carbon.emissions_kg.toFixed(3)} kg CO₂
          </span>
        </div>
      </div>
    {/if}
  </div>
</Card>
```

#### CarbonChart
```svelte
<!-- src/lib/components/CarbonChart.svelte -->
<script lang="ts">
  import { Line } from 'svelte-chartjs';
  import type { CarbonTimepoint } from '$lib/types';
  
  export let data: CarbonTimepoint[];
  
  $: chartData = {
    labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [{
      label: 'Carbon Emissions (kg CO₂)',
      data: data.map(d => d.emissions_kg),
      borderColor: 'rgb(52, 211, 153)',
      backgroundColor: 'rgba(52, 211, 153, 0.1)',
      tension: 0.4
    }]
  };
  
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };
</script>

<div class="w-full">
  <Line data={chartData} {options} />
</div>
```

---

## 4. Page Layouts

### Dashboard Layout
```svelte
<!-- src/routes/+layout.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import Header from '$lib/components/Header.svelte';
</script>

<div class="min-h-screen bg-gray-50">
  <Sidebar />
  
  <div class="lg:pl-64">
    <Header />
    
    <main class="py-10">
      <div class="px-4 sm:px-6 lg:px-8">
        <slot />
      </div>
    </main>
  </div>
</div>
```

### Dashboard Home
```svelte
<!-- src/routes/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { StatsCard, RecentJobs, CarbonSummary } from '$lib/components';
  import { api } from '$lib/api';
  
  let stats = { models: 0, jobs: 0, carbon: 0 };
  let recentJobs = [];
  
  onMount(async () => {
    stats = await api.getStats();
    recentJobs = await api.getRecentJobs();
  });
</script>

<div class="space-y-8">
  <!-- Stats -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <StatsCard title="Models" value={stats.models} icon="cube" />
    <StatsCard title="Training Jobs" value={stats.jobs} icon="cpu" />
    <StatsCard title="Carbon Saved" value="{stats.carbon} kg" icon="leaf" />
  </div>
  
  <!-- Recent Activity -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <RecentJobs jobs={recentJobs} />
    <CarbonSummary />
  </div>
</div>
```

### Models Page
```svelte
<!-- src/routes/models/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { ModelCard, Button, EmptyState } from '$lib/components';
  import { api } from '$lib/api';
  
  let models = [];
  let loading = true;
  
  onMount(async () => {
    models = await api.getModels();
    loading = false;
  });
  
  async function handleDeploy(modelId: string) {
    await api.deployModel(modelId);
  }
</script>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <h1 class="text-3xl font-bold text-gray-900">Models</h1>
    <Button href="/models/new">+ New Model</Button>
  </div>
  
  {#if loading}
    <div class="text-center py-12">Loading...</div>
  {:else if models.length === 0}
    <EmptyState 
      title="No models yet"
      description="Start by fine-tuning your first model"
      action="Train Model"
      href="/training/new"
    />
  {:else}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each models as model}
        <ModelCard 
          {model} 
          onDeploy={() => handleDeploy(model.id)}
        />
      {/each}
    </div>
  {/if}
</div>
```

### Training Job Creation
```svelte
<!-- src/routes/training/new/+page.svelte -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { Form, Select, Input, TextArea, Button } from '$lib/components';
  import { api } from '$lib/api';
  
  let formData = {
    name: '',
    base_model: '',
    dataset_id: '',
    learning_rate: 2e-4,
    num_epochs: 3,
    batch_size: 2
  };
  
  let models = [];
  let datasets = [];
  
  async function handleSubmit() {
    const job = await api.createTrainingJob(formData);
    goto(`/training/${job.job_id}`);
  }
</script>

<div class="max-w-3xl mx-auto">
  <h1 class="text-3xl font-bold mb-8">Start Training</h1>
  
  <Form on:submit={handleSubmit}>
    <Input 
      label="Model Name"
      bind:value={formData.name}
      required
      placeholder="my-finance-model"
    />
    
    <Select 
      label="Base Model"
      bind:value={formData.base_model}
      options={models}
      required
    />
    
    <Select 
      label="Dataset"
      bind:value={formData.dataset_id}
      options={datasets}
      required
    />
    
    <div class="grid grid-cols-3 gap-4">
      <Input 
        label="Learning Rate"
        type="number"
        bind:value={formData.learning_rate}
        step="0.00001"
      />
      <Input 
        label="Epochs"
        type="number"
        bind:value={formData.num_epochs}
        min="1"
      />
      <Input 
        label="Batch Size"
        type="number"
        bind:value={formData.batch_size}
        min="1"
      />
    </div>
    
    <div class="flex gap-4">
      <Button type="submit" variant="primary">Start Training</Button>
      <Button type="button" variant="ghost" href="/training">Cancel</Button>
    </div>
  </Form>
</div>
```

### Training Job Detail with Live Updates
```svelte
<!-- src/routes/training/[id]/+page.svelte -->
<script lang="ts">
  import { page } from '$app/stores';
  import { onMount, onDestroy } from 'svelte';
  import { TrainingStatus, LogViewer, Button } from '$lib/components';
  import { api } from '$lib/api';
  
  let job = null;
  let logs = [];
  let ws: WebSocket;
  
  onMount(async () => {
    const jobId = $page.params.id;
    job = await api.getJob(jobId);
    
    // Connect WebSocket for live updates
    ws = new WebSocket(`ws://localhost:8000/ws/training/${jobId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'log') {
        logs = [...logs, data];
      } else if (data.type === 'metrics') {
        job.metrics = data.metrics;
      } else if (data.type === 'status_change') {
        job.status = data.new_status;
      }
    };
  });
  
  onDestroy(() => {
    ws?.close();
  });
  
  async function handleCancel() {
    await api.cancelJob(job.job_id);
  }
</script>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <h1 class="text-3xl font-bold">Training Job</h1>
    {#if job?.status === 'running'}
      <Button variant="danger" on:click={handleCancel}>
        Cancel Training
      </Button>
    {/if}
  </div>
  
  {#if job}
    <TrainingStatus {job} />
    
    <Card>
      <h2 class="text-lg font-semibold mb-4">Training Logs</h2>
      <LogViewer {logs} />
    </Card>
  {/if}
</div>
```

---

## 5. State Management

### Stores
```typescript
// src/lib/stores/models.ts
import { writable } from 'svelte/store';
import type { Model } from '$lib/types';

export const models = writable<Model[]>([]);
export const selectedModel = writable<Model | null>(null);

export async function loadModels() {
  const response = await fetch('/api/v1/models');
  const data = await response.json();
  models.set(data.items);
}
```

```typescript
// src/lib/stores/auth.ts (Phase 2+)
import { writable } from 'svelte/store';

interface User {
  id: string;
  email: string;
  name: string;
}

export const user = writable<User | null>(null);
export const isAuthenticated = writable(false);
```

---

## 6. API Client

```typescript
// src/lib/api/client.ts
const API_BASE = import.meta.env.VITE_API_BASE_URL;

class APIClient {
  private baseURL: string;
  
  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }
  
  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  // Models
  async getModels() {
    return this.request('/models');
  }
  
  async getModel(id: string) {
    return this.request(`/models/${id}`);
  }
  
  async deleteModel(id: string) {
    return this.request(`/models/${id}`, { method: 'DELETE' });
  }
  
  // Training
  async createTrainingJob(data: any) {
    return this.request('/training/jobs', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
  
  async getJob(id: string) {
    return this.request(`/training/jobs/${id}`);
  }
  
  async cancelJob(id: string) {
    return this.request(`/training/jobs/${id}/cancel`, {
      method: 'POST',
    });
  }
  
  // Inference
  async generate(data: any) {
    return this.request('/inference/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

export const api = new APIClient(API_BASE);
```

---

## 7. Responsive Design

### Breakpoints (Tailwind defaults)
- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

### Mobile-First Approach
```svelte
<div class="
  grid grid-cols-1 gap-4
  md:grid-cols-2
  lg:grid-cols-3
  xl:grid-cols-4
">
  <!-- Cards -->
</div>
```

---

## 8. Accessibility

### ARIA Labels
```svelte
<button aria-label="Delete model">
  <TrashIcon />
</button>
```

### Keyboard Navigation
- Ensure all interactive elements are keyboard accessible
- Tab order should be logical
- Escape key should close modals
- Enter/Space should activate buttons

### Color Contrast
- Ensure text has sufficient contrast (WCAG AA minimum)
- Don't rely solely on color for information

---

## 9. Performance Optimization

### Code Splitting
```typescript
// Dynamic imports for heavy components
const Chart = () => import('$lib/components/Chart.svelte');
```

### Lazy Loading
```svelte
{#await import('$lib/components/HeavyComponent.svelte')}
  <Spinner />
{:then Component}
  <Component.default />
{/await}
```

### Image Optimization
```svelte
<img
  src="/images/model.jpg"
  alt="Model diagram"
  loading="lazy"
  width="400"
  height="300"
/>
```

---

## 10. Error Handling

```svelte
<script lang="ts">
  import { ErrorBoundary, Toast } from '$lib/components';
  
  let showError = false;
  let errorMessage = '';
  
  async function handleAction() {
    try {
      await api.doSomething();
    } catch (error) {
      errorMessage = error.message;
      showError = true;
    }
  }
</script>

{#if showError}
  <Toast type="error" message={errorMessage} />
{/if}
```

---

## 11. Testing

```typescript
// src/lib/components/__tests__/Button.test.ts
import { render, fireEvent } from '@testing-library/svelte';
import Button from '../Button.svelte';

test('button click handler', async () => {
  const { getByRole } = render(Button, {
    props: { variant: 'primary' }
  });
  
  const button = getByRole('button');
  await fireEvent.click(button);
  
  // Assert behavior
});
```

---

## 12. Build Configuration

```typescript
// vite.config.ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
});
```

```javascript
// tailwind.config.js
module.exports = {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          600: '#2563eb',
        },
        carbon: {
          500: '#34d399',
          600: '#10b981',
        }
      }
    }
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography')
  ]
};
```

---

## Next Steps
1. Set up SvelteKit project
2. Configure TailwindCSS
3. Create base component library
4. Implement layouts and navigation
5. Build dashboard pages
6. Integrate with backend API
7. Add real-time WebSocket updates
8. Implement carbon tracking visualizations
