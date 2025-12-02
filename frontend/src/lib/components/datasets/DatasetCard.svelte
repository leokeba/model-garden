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
</script>

<Card class="hover:shadow-lg transition-shadow">
    <div class="p-6">
        <!-- Header -->
        <div class="flex justify-between items-start mb-4">
            <div class="flex-1 min-w-0">
                <h3 class="text-lg font-semibold text-gray-900 truncate mb-1">
                    {dataset.name}
                </h3>
                <Badge variant={getFormatBadgeColor(dataset.format)} size="sm">
                    {dataset.format.toUpperCase()}
                </Badge>
            </div>
            <button
                onclick={() => onDelete(dataset.name)}
                class="text-gray-400 hover:text-red-600 transition-colors ml-2"
                title="Delete dataset"
            >
                🗑️
            </button>
        </div>

        <!-- Stats -->
        <div class="space-y-2 mb-4">
            <div class="flex justify-between text-sm">
                <span class="text-gray-600">Examples:</span>
                <span class="font-medium text-gray-900">
                    {dataset.examples.toLocaleString()}
                </span>
            </div>
            <div class="flex justify-between text-sm">
                <span class="text-gray-600">Size:</span>
                <span class="font-medium text-gray-900">
                    {formatFileSize(dataset.size)}
                </span>
            </div>
            <div class="flex justify-between text-sm">
                <span class="text-gray-600">Created:</span>
                <span
                    class="font-medium text-gray-900 truncate ml-2"
                    title={formatDate(dataset.created_at)}
                >
                    {new Date(dataset.created_at).toLocaleDateString()}
                </span>
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
