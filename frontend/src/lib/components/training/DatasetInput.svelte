<script lang="ts">
    interface Props {
        datasetPath: string;
        fromHub: boolean;
        modelType: "text" | "vision";
        onDatasetPathChange: (value: string) => void;
        onFromHubChange: (value: boolean) => void;
    }

    let {
        datasetPath,
        fromHub,
        modelType,
        onDatasetPathChange,
        onFromHubChange,
    }: Props = $props();
</script>

<div>
    <label
        for="dataset_path"
        class="block text-sm font-medium text-gray-700 mb-1"
    >
        Dataset Path *
    </label>
    <input
        type="text"
        id="dataset_path"
        value={datasetPath}
        oninput={(e) => onDatasetPathChange(e.currentTarget.value)}
        placeholder={fromHub
            ? "username/dataset-name"
            : modelType === "vision"
              ? "./data/vision_dataset.jsonl"
              : "./data/my-dataset.jsonl"}
        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        required
    />
    <div class="mt-2 flex items-center">
        <input
            type="checkbox"
            id="from_hub"
            checked={fromHub}
            onchange={(e) => onFromHubChange(e.currentTarget.checked)}
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
        />
        <label for="from_hub" class="ml-2 block text-sm text-gray-700">
            Load from HuggingFace Hub
        </label>
    </div>
    <p class="text-xs text-gray-500 mt-1">
        {#if fromHub}
            Enter a HuggingFace dataset identifier (e.g.,
            "username/dataset-name")<br />
            For specific files, use: "username/repo::train.jsonl"
        {:else if modelType === "vision"}
            Path to your JSONL dataset with image paths/base64 or local file
        {:else}
            Path to your JSONL dataset file
        {/if}
    </p>
</div>
