<script lang="ts">
    import Badge from "$lib/components/Badge.svelte";
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
        dataset: Dataset;
        onDelete: (name: string) => void;
        onPreview: (dataset: Dataset) => void;
    }

    let { dataset, onDelete, onPreview }: Props = $props();

    function formatFileSize(bytes: number): string {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    function formatDate(dateString: string): string {
        if (!dateString) return "N/A";
        return new Date(dateString).toLocaleString();
    }

    function getFormatBadgeColor(
        format: string,
    ): "success" | "info" | "warning" {
        const formats: Record<string, "success" | "info" | "warning"> = {
            jsonl: "success",
            json: "success",
            csv: "info",
            parquet: "info",
            txt: "warning",
        };
        return formats[format] || "info";
    }

    function getDatasetType(dataset: Dataset): string {
        const metaType = (dataset.metadata?.type || dataset.metadata?.modality || "")
            .toString()
            .toLowerCase();

        if (metaType.includes("vision") || metaType.includes("image")) return "Vision";
        if (metaType.includes("multi")) return "Multimodal";
        if (metaType.includes("audio")) return "Audio";
        if (metaType) return metaType.charAt(0).toUpperCase() + metaType.slice(1);
        return "Text";
    }

    const datasetType = $derived(getDatasetType(dataset));
    const updatedAt = $derived(dataset.modified_at || dataset.created_at);
</script>

<Card class="group hover:shadow-xl transition-all duration-200 border border-gray-100 hover:-translate-y-[2px]">
    <div class="p-6 space-y-4">
        <!-- Header -->
        <div class="flex justify-between items-start gap-3">
            <div class="flex-1 min-w-0 space-y-2">
                <div class="flex items-center gap-2 flex-wrap">
                    <span
                        class="px-3 py-1 text-xs font-semibold rounded-full border {datasetType.toLowerCase().includes('vision')
                            ? 'bg-primary-50 text-primary-700 border-primary-100'
                            : 'bg-gray-100 text-gray-700 border-gray-200'}"
                    >
                        {datasetType}
                    </span>
                    <Badge variant={getFormatBadgeColor(dataset.format)} size="sm">
                        {dataset.format.toUpperCase()}
                    </Badge>
                    {#if dataset.metadata?.split}
                        <Badge variant="info" size="sm">{dataset.metadata.split}</Badge>
                    {/if}
                </div>
                <h3 class="text-lg font-semibold text-gray-900 truncate">
                    {dataset.name}
                </h3>
                <p class="text-xs text-gray-500 truncate" title={dataset.path}>
                    {dataset.path}
                </p>
            </div>
            <div class="flex items-center gap-2">
                <button
                    onclick={() => onPreview(dataset)}
                    class="px-3 py-2 text-sm rounded-lg bg-gray-100 text-gray-700 hover:bg-primary-50 hover:text-primary-700 transition-colors"
                    title="Quick preview"
                >
                    👁️
                </button>
                <button
                    onclick={() => onDelete(dataset.name)}
                    class="text-gray-400 hover:text-red-600 transition-colors"
                    title="Delete dataset"
                >
                    🗑️
                </button>
            </div>
        </div>

        <!-- Stats -->
        <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="rounded-xl bg-gray-50 px-3 py-2 border border-gray-100">
                <p class="text-gray-500">Examples</p>
                <p class="font-semibold text-gray-900">
                    {dataset.examples.toLocaleString()}
                </p>
            </div>
            <div class="rounded-xl bg-gray-50 px-3 py-2 border border-gray-100">
                <p class="text-gray-500">Size</p>
                <p class="font-semibold text-gray-900">{formatFileSize(dataset.size)}</p>
            </div>
            <div class="rounded-xl bg-gray-50 px-3 py-2 border border-gray-100">
                <p class="text-gray-500">Updated</p>
                <p class="font-semibold text-gray-900" title={updatedAt ? formatDate(updatedAt) : ''}>
                    {updatedAt ? new Date(updatedAt).toLocaleDateString() : '—'}
                </p>
            </div>
            <div class="rounded-xl bg-gray-50 px-3 py-2 border border-gray-100">
                <p class="text-gray-500">Path</p>
                <p class="font-semibold text-gray-900 truncate" title={dataset.path}>
                    {dataset.path.split('/').slice(-2).join('/') || dataset.name}
                </p>
            </div>
        </div>

        <!-- Actions -->
        <div class="flex gap-2">
            <Button
                onclick={() => onPreview(dataset)}
                variant="secondary"
                size="sm"
                fullWidth
            >
                👁️ Preview
            </Button>
            <Button
                href={`/training/new?dataset=${dataset.name}`}
                variant="primary"
                size="sm"
                fullWidth
            >
                🎓 Train
            </Button>
        </div>
    </div>
</Card>
