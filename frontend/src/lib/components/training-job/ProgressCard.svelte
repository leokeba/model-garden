<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
    }

    let { job }: Props = $props();

    function getProgressPercent(job: TrainingJob): number {
        if (job.status === "completed") return 100;

        // Try to use progress object
        if (job.progress) {
            if (typeof job.progress === "number") {
                return Math.round(job.progress * 100);
            }
            if (
                typeof job.progress.current_step === "number" &&
                typeof job.progress.total_steps === "number" &&
                job.progress.total_steps > 0
            ) {
                return Math.round(
                    (job.progress.current_step / job.progress.total_steps) *
                        100,
                );
            }
        }

        // Try to use top-level properties
        if (
            typeof job.current_step === "number" &&
            typeof job.total_steps === "number" &&
            job.total_steps > 0
        ) {
            return Math.round((job.current_step / job.total_steps) * 100);
        }

        return 0;
    }

    let progressPercent = $derived(getProgressPercent(job));

    function formatDuration(seconds: number): string {
        if (seconds === undefined || seconds === null || seconds < 0)
            return "calculating...";

        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);

        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
    }
</script>

{#if job.status !== "queued" && job.status !== "pending"}
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
                        <span>{progressPercent}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div
                            class="bg-primary-600 h-2 rounded-full transition-all duration-300"
                            style="width: {progressPercent}%"
                        ></div>
                    </div>
                </div>

                {#if job.current_step !== undefined && job.total_steps !== undefined}
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

                {#if job.current_epoch !== undefined}
                    <div class="text-sm">
                        <dt class="block text-gray-700">Current Epoch</dt>
                        <dd class="font-semibold">{job.current_epoch}</dd>
                    </div>
                {/if}

                {#if job.progress?.eta_seconds !== undefined && job.status === "running"}
                    <div
                        class="grid grid-cols-2 gap-4 text-sm pt-2 border-t border-gray-100"
                    >
                        <div>
                            <dt class="block text-gray-700">
                                Estimated Time Remaining
                            </dt>
                            <dd class="font-semibold text-primary-700">
                                {formatDuration(job.progress.eta_seconds)}
                            </dd>
                        </div>
                        <div>
                            <dt class="block text-gray-700">Speed</dt>
                            <dd class="font-semibold">
                                {job.progress.steps_per_second
                                    ? job.progress.steps_per_second.toFixed(2)
                                    : "0.00"} steps/s
                            </dd>
                        </div>
                    </div>
                {/if}
            </div>
        </div>
    </Card>
{/if}
