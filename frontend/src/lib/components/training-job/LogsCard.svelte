<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
        logs: string[];
        isConnected: boolean;
        logsContainer?: HTMLDivElement | null;
    }

    let {
        job,
        logs,
        isConnected,
        logsContainer = $bindable(),
    }: Props = $props();
</script>

<!-- Real-time Logs -->
{#if (job.status === "running" || job.status === "queued") && logs.length > 0}
    <Card>
        <div class="p-6">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-semibold text-gray-900">
                    Real-time Logs
                </h2>
                {#if isConnected}
                    <span class="text-xs text-gray-500">Live</span>
                {/if}
            </div>

            <div
                bind:this={logsContainer}
                class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto max-h-96 text-sm font-mono"
            >
                {#each logs as log}
                    <div class="mb-1">{log}</div>
                {/each}
            </div>
        </div>
    </Card>
{/if}

<!-- Historical Logs -->
{#if job.logs && job.logs.length > 0}
    <Card>
        <div class="p-6">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">
                Training Logs
            </h2>

            <div
                class="bg-gray-900 text-green-400 p-4 rounded-lg overflow-auto max-h-96 text-sm font-mono"
            >
                {#each job.logs as log}
                    <div class="mb-1">{log}</div>
                {/each}
            </div>
        </div>
    </Card>
{/if}
