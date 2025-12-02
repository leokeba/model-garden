<script lang="ts">
    import Badge from "$lib/components/Badge.svelte";
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    type HubDataset = {
        id: string;
        author: string;
        datasetName: string;
        downloads: number;
        likes: number;
        tags: string[];
        description?: string;
        size?: string;
    };

    interface Props {
        dataset: HubDataset;
        loading: boolean;
        onLoad: (datasetId: string) => void;
    }

    let { dataset, loading, onLoad }: Props = $props();
</script>

<Card class="hover:shadow-lg transition-shadow">
    <div class="p-6">
        <!-- Header -->
        <div class="mb-4">
            <div class="flex items-start justify-between mb-2">
                <div class="flex-1 min-w-0">
                    <h3
                        class="text-lg font-semibold text-gray-900 mb-1 truncate"
                        title={dataset.id}
                    >
                        {dataset.datasetName}
                    </h3>
                    <p class="text-sm text-gray-600 truncate">
                        by <span class="font-medium">{dataset.author}</span>
                    </p>
                </div>
            </div>

            {#if dataset.description}
                <p class="text-sm text-gray-700 line-clamp-2 mb-3">
                    {dataset.description}
                </p>
            {/if}

            <!-- Tags -->
            <div class="flex flex-wrap gap-1 mb-3">
                {#each dataset.tags.slice(0, 3) as tag}
                    <Badge variant="info" size="sm">
                        {tag}
                    </Badge>
                {/each}
            </div>
        </div>

        <!-- Stats -->
        <div
            class="flex items-center gap-4 text-sm text-gray-600 mb-4 pb-4 border-b border-gray-200"
        >
            <div class="flex items-center gap-1" title="Downloads">
                <span>⬇️</span>
                <span>{(dataset.downloads / 1000).toFixed(0)}k</span>
            </div>
            <div class="flex items-center gap-1" title="Likes">
                <span>❤️</span>
                <span>{dataset.likes}</span>
            </div>
            {#if dataset.size}
                <div class="flex items-center gap-1" title="Size">
                    <span>📦</span>
                    <span>{dataset.size}</span>
                </div>
            {/if}
        </div>

        <!-- Actions -->
        <div class="flex gap-2">
            <Button
                onclick={() =>
                    window.open(
                        `https://huggingface.co/datasets/${dataset.id}`,
                        "_blank",
                    )}
                variant="secondary"
                size="sm"
                fullWidth
            >
                🤗 View on Hub
            </Button>
            <Button
                onclick={() => onLoad(dataset.id)}
                variant="primary"
                size="sm"
                fullWidth
                {loading}
                disabled={loading}
            >
                💾 Load
            </Button>
        </div>
    </div>
</Card>
