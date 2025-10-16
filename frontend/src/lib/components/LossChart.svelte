<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Chart, registerables } from 'chart.js';
  
  Chart.register(...registerables);
  
  interface MetricPoint {
    step: number;
    loss: number;
    timestamp: string;
    [key: string]: any;
  }
  
  interface Props {
    trainingMetrics?: MetricPoint[];
    validationMetrics?: MetricPoint[];
    title?: string;
    height?: number;
  }
  
  let { 
    trainingMetrics = [],
    validationMetrics = [],
    title = 'Training & Validation Loss',
    height = 300
  }: Props = $props();
  
  let canvas: HTMLCanvasElement | null = $state(null);
  let chart: Chart | null = null;
  
  function createChart() {
    if (!canvas) return;
    
    // Destroy existing chart
    if (chart) {
      chart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Prepare training data
    const trainData = trainingMetrics.map(m => ({
      x: m.step,
      y: m.loss
    }));
    
    // Prepare validation data
    const valData = validationMetrics.map(m => ({
      x: m.step,
      y: m.loss
    }));
    
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Training Loss',
            data: trainData,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            fill: true,
            tension: 0.4,
          },
          ...(valData.length > 0 ? [{
            label: 'Validation Loss',
            data: valData,
            borderColor: 'rgb(16, 185, 129)',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            fill: true,
            tension: 0.4,
          }] : [])
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
          },
          title: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  label += context.parsed.y.toFixed(4);
                }
                return label;
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Training Steps'
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Loss'
            },
            beginAtZero: false,
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          }
        }
      }
    });
  }
  
  // Update chart when metrics change
  $effect(() => {
    if (canvas && (trainingMetrics.length > 0 || validationMetrics.length > 0)) {
      createChart();
    }
  });
  
  onMount(() => {
    if (canvas) {
      createChart();
    }
  });
  
  onDestroy(() => {
    if (chart) {
      chart.destroy();
    }
  });
</script>

<div class="w-full">
  <h3 class="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
  
  {#if trainingMetrics.length === 0 && validationMetrics.length === 0}
    <div class="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
      <p class="text-gray-500">No metrics data available yet</p>
    </div>
  {:else}
    <div class="relative" style="height: {height}px;">
      <canvas bind:this={canvas}></canvas>
    </div>
    
    <!-- Metrics Summary -->
    <div class="mt-4 grid grid-cols-2 gap-4 text-sm">
      {#if trainingMetrics.length > 0}
        <div class="p-3 bg-blue-50 rounded-lg">
          <div class="text-xs font-medium text-blue-700 mb-1">Training Loss</div>
          <div class="flex items-baseline gap-2">
            <span class="text-lg font-bold text-blue-900">
              {trainingMetrics[trainingMetrics.length - 1].loss.toFixed(4)}
            </span>
            <span class="text-xs text-blue-600">
              Latest ({trainingMetrics.length} points)
            </span>
          </div>
        </div>
      {/if}
      
      {#if validationMetrics.length > 0}
        <div class="p-3 bg-green-50 rounded-lg">
          <div class="text-xs font-medium text-green-700 mb-1">Validation Loss</div>
          <div class="flex items-baseline gap-2">
            <span class="text-lg font-bold text-green-900">
              {validationMetrics[validationMetrics.length - 1].loss.toFixed(4)}
            </span>
            <span class="text-xs text-green-600">
              Latest ({validationMetrics.length} points)
            </span>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>
