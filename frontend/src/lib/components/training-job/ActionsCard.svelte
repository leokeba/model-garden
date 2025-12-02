<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
        cancelling: boolean;
        rerunning: boolean;
        onRefresh: () => void;
        onCancel: () => void;
        onRerun: () => void;
    }

    let { job, cancelling, rerunning, onRefresh, onCancel, onRerun }: Props =
        $props();
</script>

<Card>
    <div class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Actions</h3>

        <div class="space-y-2">
            <Button variant="primary" fullWidth onclick={onRefresh}>
                Refresh Status
            </Button>

            {#if job.status === "running" || job.status === "queued"}
                <Button
                    variant="danger"
                    fullWidth
                    onclick={onCancel}
                    loading={cancelling}
                    disabled={cancelling}
                >
                    {cancelling ? "Cancelling..." : "Cancel Training"}
                </Button>
            {/if}

            {#if job.status === "completed" || job.status === "failed" || job.status === "cancelled"}
                <Button
                    variant="primary"
                    fullWidth
                    onclick={onRerun}
                    loading={rerunning}
                    disabled={rerunning}
                >
                    {rerunning ? "Starting Rerun..." : "🔄 Rerun Training"}
                </Button>
            {/if}

            {#if job.status === "completed" && job.config}
                <Button
                    href="/models/{job.config.name}"
                    variant="secondary"
                    fullWidth
                >
                    View Model
                </Button>
            {/if}

            <Button href="/training/new" variant="secondary" fullWidth>
                Start New Job
            </Button>
        </div>
    </div>
</Card>
