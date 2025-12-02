<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
    }

    let { job }: Props = $props();

    function formatProgress(progress: any) {
        if (typeof progress === "number") {
            return Math.round(progress * 100);
        }
        if (
            progress &&
            typeof progress.current_step === "number" &&
            typeof progress.total_steps === "number"
        ) {
            if (progress.total_steps === 0) {
                return 0;
            }
            return Math.round(
                (progress.current_step / progress.total_steps) * 100,
            );
        }
        return 0;
    }
</script>

{#if job.status === "running" && job.progress}
    <Card>
        <div class="p-6">
            <h2 class="text-xl font-semibold text-gray-900 mb-4">
                Training Progress
            </h2>

            <div class="space-y-4">
                <div>
                    <div
                        class="flex justify-between text-sm text-gray-700 mb-1"
                    >
                        <span>Progress</span>
                        <span>{formatProgress(job.progress)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div
                            class="bg-primary-600 h-2 rounded-full transition-all duration-300"
                            style="width: {formatProgress(job.progress)}%"
                        ></div>
                    </div>
                </div>

                {#if job.current_step && job.total_steps}
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <dt class="block text-gray-700">Current Step</dt>
                            <dd class="font-semibold">{job.current_step}</dd>
                        </div>
                        <div>
                            <dt class="block text-gray-700">Total Steps</dt>
                            <dd class="font-semibold">{job.total_steps}</dd>
                        </div>
                    </div>
                {/if}

                {#if job.current_epoch}
                    <div class="text-sm">
                        <dt class="block text-gray-700">Current Epoch</dt>
                        <dd class="font-semibold">{job.current_epoch}</dd>
                    </div>
                {/if}
            </div>
        </div>
    </Card>
{/if}
