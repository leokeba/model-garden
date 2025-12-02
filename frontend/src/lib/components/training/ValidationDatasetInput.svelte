<script lang="ts">
    interface Props {
        validationDatasetPath: string;
        validationFromHub: boolean;
        modelType: "text" | "vision";
        onValidationDatasetPathChange: (value: string) => void;
        onValidationFromHubChange: (value: boolean) => void;
    }

    let {
        validationDatasetPath,
        validationFromHub,
        modelType,
        onValidationDatasetPathChange,
        onValidationFromHubChange,
    }: Props = $props();
</script>

<div>
    <label
        for="validation_dataset_path"
        class="block text-sm font-medium text-gray-700 mb-1"
    >
        Validation Dataset Path (Optional)
    </label>
    <input
        type="text"
        id="validation_dataset_path"
        value={validationDatasetPath}
        oninput={(e) => onValidationDatasetPathChange(e.currentTarget.value)}
        placeholder={validationFromHub
            ? "username/val-dataset-name"
            : modelType === "vision"
              ? "./data/vision_val_dataset.jsonl"
              : "./data/my-val-dataset.jsonl"}
        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
    />
    <div class="mt-2 flex items-center">
        <input
            type="checkbox"
            id="validation_from_hub"
            checked={validationFromHub}
            onchange={(e) => onValidationFromHubChange(e.currentTarget.checked)}
            class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
        />
        <label
            for="validation_from_hub"
            class="ml-2 block text-sm text-gray-700"
        >
            Load validation dataset from HuggingFace Hub
        </label>
    </div>
    <p class="text-xs text-gray-500 mt-1">
        📊 Optional: Provide a validation dataset to track validation loss
        during training<br />
        {#if validationFromHub}
            Use HuggingFace format: "username/repo" or
            "username/repo::validation.jsonl"
        {/if}
    </p>
</div>
