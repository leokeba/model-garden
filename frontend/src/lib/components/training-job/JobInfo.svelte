<script lang="ts">
    import type { TrainingJob } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        job: TrainingJob;
    }

    let { job }: Props = $props();

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

    function formatDate(dateString: string) {
        return new Date(dateString).toLocaleString();
    }
</script>

<Card>
    <div class="p-6">
        <h2 class="text-xl font-semibold text-gray-900 mb-4">
            Job Information
        </h2>

        <div class="grid grid-cols-2 gap-4">
            <div>
                <dt class="text-sm font-medium text-gray-700">Job ID</dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {job.job_id || job.id}
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">Status</dt>
                <dd class="mt-1">
                    <span
                        class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getStatusColor(
                            job.status,
                        )}"
                    >
                        {job.status}
                    </span>
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">Model Name</dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.name || job.name}
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">Base Model</dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.base_model || job.base_model}
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">Dataset</dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.dataset_path || job.dataset_path}
                    {#if job.dataset_num_samples}
                        <span class="text-xs text-gray-500 block">
                            {job.dataset_num_samples.toLocaleString()} samples
                        </span>
                    {/if}
                    {#if job.dataset_size}
                        <span class="text-xs text-gray-500 block">
                            {(job.dataset_size / 1024 / 1024).toFixed(2)} MB
                        </span>
                    {/if}
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">
                    Output Directory
                </dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {job.config?.output_dir || job.output_dir}
                </dd>
            </div>

            <div>
                <dt class="text-sm font-medium text-gray-700">Created</dt>
                <dd class="mt-1 text-sm text-gray-900">
                    {formatDate(job.created_at)}
                </dd>
            </div>

            {#if job.completed_at}
                <div>
                    <dt class="text-sm font-medium text-gray-700">Completed</dt>
                    <dd class="mt-1 text-sm text-gray-900">
                        {formatDate(job.completed_at)}
                    </dd>
                </div>
            {/if}
        </div>
    </div>
</Card>
