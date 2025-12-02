<script lang="ts">
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    type Dataset = {
        name: string;
        path: string;
        size: number;
        examples: number;
        format: string;
        created_at: string;
        modified_at?: string;
        metadata?: Record<string, any>;
    };

    interface Props {
        dataset: Dataset | null;
        previewData: any[];
        loading: boolean;
        onClose: () => void;
    }

    let { dataset, previewData, loading, onClose }: Props = $props();
</script>

{#if dataset}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    >
        <Card class="max-w-4xl w-full max-h-[80vh] flex flex-col">
            <div class="p-6 border-b border-gray-200">
                <div class="flex justify-between items-center">
                    <div>
                        <h2 class="text-2xl font-bold text-gray-900">
                            {dataset.name}
                        </h2>
                        <p class="text-sm text-gray-600 mt-1">
                            Showing first 10 samples
                        </p>
                    </div>
                    <button
                        onclick={onClose}
                        class="text-gray-400 hover:text-gray-600"
                    >
                        ✕
                    </button>
                </div>
            </div>

            <div class="flex-1 overflow-y-auto p-6">
                {#if loading}
                    <div class="flex justify-center items-center h-32">
                        <div
                            class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                        ></div>
                    </div>
                {:else if previewData.length === 0}
                    <p class="text-center text-gray-500 py-8">
                        No data to preview
                    </p>
                {:else}
                    <div class="space-y-4">
                        {#each previewData as sample, index}
                            <div
                                class="border border-gray-200 rounded-lg p-4 bg-gray-50"
                            >
                                <div
                                    class="text-xs font-medium text-gray-500 mb-2"
                                >
                                    Sample {index + 1}
                                </div>
                                <pre
                                    class="text-sm text-gray-900 whitespace-pre-wrap overflow-x-auto">{JSON.stringify(
                                        sample,
                                        null,
                                        2,
                                    )}</pre>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>

            <div class="p-6 border-t border-gray-200">
                <Button onclick={onClose} variant="secondary" fullWidth>
                    Close
                </Button>
            </div>
        </Card>
    </div>
{/if}
