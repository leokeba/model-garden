<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Button from "$lib/components/Button.svelte";

    interface Props {
        job: TrainingJob | null;
        isConnected: boolean;
        cancelling: boolean;
        stoppingEarly: boolean;
        onCancel: () => void;
        onStopEarly: () => void;
        onBack?: () => void;
    }

    let {
        job,
        isConnected,
        cancelling,
        stoppingEarly,
        onCancel,
        onStopEarly,
        onBack,
    }: Props = $props();

    function getStatusColor(status: string) {
        switch (status) {
            case "running":
                return "text-blue-600 bg-blue-100";
            case "completed":
                return "text-green-600 bg-green-100";
            case "failed":
                return "text-red-600 bg-red-100";
            case "cancelled":
                return "text-gray-600 bg-gray-100";
            case "queued":
                return "text-yellow-600 bg-yellow-100";
            default:
                return "text-gray-600 bg-gray-100";
        }
    }
</script>

<div class="mb-6">
    <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
            <Button href="/training" variant="ghost" size="sm">← Back</Button>
            <div>
                <h1 class="text-3xl font-bold text-gray-900">
                    Training Job Details
                </h1>
                <p class="mt-1 text-sm text-gray-600">
                    Monitor and manage your training job
                </p>
            </div>
        </div>
        {#if job}
            <div class="flex items-center gap-3">
                {#if job.status === "running"}
                    <Button
                        variant="warning"
                        size="sm"
                        onclick={onStopEarly}
                        loading={stoppingEarly}
                        disabled={stoppingEarly || cancelling}
                    >
                        {stoppingEarly ? "Stopping..." : "⏸️ Stop Early"}
                    </Button>
                {/if}
                {#if job.status === "running" || job.status === "queued"}
                    <Button
                        variant="danger"
                        size="sm"
                        onclick={onCancel}
                        loading={cancelling}
                        disabled={cancelling || stoppingEarly}
                    >
                        {cancelling ? "Cancelling..." : "Cancel"}
                    </Button>
                {/if}

                {#if job.status === "running"}
                    {#if isConnected}
                        <span
                            class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-green-600 bg-green-100"
                        >
                            <span
                                class="w-2 h-2 bg-green-600 rounded-full mr-1.5 animate-pulse"
                            ></span>
                            Live Updates
                        </span>
                    {:else}
                        <span
                            class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-yellow-600 bg-yellow-100"
                        >
                            <span
                                class="w-2 h-2 bg-yellow-600 rounded-full mr-1.5"
                            ></span>
                            Reconnecting...
                        </span>
                    {/if}
                {/if}
                <span
                    class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(
                        job.status,
                    )}"
                >
                    {job.status}
                </span>
            </div>
        {/if}
    </div>
</div>
