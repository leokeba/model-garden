<script lang="ts">
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        show: boolean;
        uploading: boolean;
        uploadProgress: number;
        onClose: () => void;
        onUpload: () => void;
        selectedFile: File | null;
        datasetName: string;
        datasetType: string;
        onFileSelect: (e: Event) => void;
        onNameChange: (name: string) => void;
        onTypeChange: (type: string) => void;
    }

    let {
        show,
        uploading,
        uploadProgress,
        onClose,
        onUpload,
        selectedFile,
        datasetName,
        datasetType,
        onFileSelect,
        onNameChange,
        onTypeChange,
    }: Props = $props();

    let fileInput: HTMLInputElement | undefined = $state(undefined);
</script>

{#if show}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    >
        <Card class="max-w-2xl w-full">
            <div class="p-6">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold text-gray-900">
                        Upload Dataset
                    </h2>
                    <button
                        onclick={onClose}
                        class="text-gray-400 hover:text-gray-600"
                        disabled={uploading}
                    >
                        ✕
                    </button>
                </div>

                <div class="space-y-6">
                    <!-- File Input -->
                    <div>
                        <label
                            for="dataset-file"
                            class="block text-sm font-medium text-gray-700 mb-2"
                        >
                            Dataset File
                        </label>
                        <input
                            type="file"
                            id="dataset-file"
                            bind:this={fileInput}
                            onchange={onFileSelect}
                            accept=".json,.jsonl,.csv,.txt,.parquet"
                            disabled={uploading}
                            class="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-lg file:border-0
                file:text-sm file:font-semibold
                file:bg-primary-50 file:text-primary-700
                hover:file:bg-primary-100
                disabled:opacity-50 disabled:cursor-not-allowed"
                        />
                        <p class="mt-2 text-xs text-gray-500">
                            Supported formats: JSON, JSONL, CSV, TXT, Parquet
                        </p>
                    </div>

                    <!-- Dataset Name -->
                    <div>
                        <label
                            for="dataset-name"
                            class="block text-sm font-medium text-gray-700 mb-2"
                        >
                            Dataset Name
                        </label>
                        <input
                            type="text"
                            id="dataset-name"
                            value={datasetName}
                            oninput={(e) => onNameChange(e.currentTarget.value)}
                            placeholder="my-dataset"
                            disabled={uploading}
                            class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100"
                        />
                    </div>

                    <!-- Dataset Type -->
                    <div>
                        <label
                            for="dataset-type"
                            class="block text-sm font-medium text-gray-700 mb-2"
                        >
                            Dataset Type
                        </label>
                        <select
                            id="dataset-type"
                            value={datasetType}
                            onchange={(e) =>
                                onTypeChange(e.currentTarget.value)}
                            disabled={uploading}
                            class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:bg-gray-100"
                        >
                            <option value="text">Text</option>
                            <option value="vision">Vision</option>
                            <option value="multimodal">Multimodal</option>
                        </select>
                    </div>

                    <!-- Upload Progress -->
                    {#if uploading}
                        <div>
                            <div
                                class="flex justify-between text-sm text-gray-600 mb-2"
                            >
                                <span>Uploading...</span>
                                <span>{Math.round(uploadProgress)}%</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div
                                    class="bg-primary-600 h-2 rounded-full transition-all duration-300"
                                    style="width: {uploadProgress}%"
                                ></div>
                            </div>
                        </div>
                    {/if}

                    <!-- Actions -->
                    <div class="flex gap-3 pt-4">
                        <Button
                            onclick={onClose}
                            variant="secondary"
                            fullWidth
                            disabled={uploading}
                        >
                            Cancel
                        </Button>
                        <Button
                            onclick={onUpload}
                            variant="primary"
                            fullWidth
                            disabled={!selectedFile ||
                                !datasetName.trim() ||
                                uploading}
                            loading={uploading}
                        >
                            {uploading ? "Uploading..." : "Upload"}
                        </Button>
                    </div>
                </div>
            </div>
        </Card>
    </div>
{/if}
