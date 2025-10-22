<script lang="ts">
    import { api, type Model } from "$lib/api/client";
    import Button from "./Button.svelte";

    interface Props {
        model: Model;
        isOpen: boolean;
        onClose: () => void;
        onSuccess?: (url: string) => void;
    }

    let {
        model,
        isOpen = $bindable(false),
        onClose,
        onSuccess,
    }: Props = $props();

    let repoId = $state("");
    let isPrivate = $state(false);
    let commitMessage = $state("Upload model from Model Garden");
    let repoDescription = $state(
        `Model fine-tuned with Model Garden. Base model: ${model.base_model}`,
    );
    let uploading = $state(false);
    let error = $state("");
    let uploadSuccess = $state(false);
    let uploadedUrl = $state("");

    // Pre-fill repo_id with a sensible default
    $effect(() => {
        if (isOpen && !repoId) {
            // Generate default repo name from model name (sanitized)
            const sanitizedName = model.name
                .toLowerCase()
                .replace(/[^a-z0-9-_]/g, "-")
                .replace(/-+/g, "-")
                .replace(/^-|-$/g, "");
            repoId = `username/${sanitizedName}`;
        }
    });

    async function handleUpload() {
        error = "";

        // Validate repo_id format
        if (!repoId || !repoId.includes("/")) {
            error = 'Repository ID must be in the format "username/repo-name"';
            return;
        }

        uploading = true;

        try {
            const response = await api.uploadModelToHub(model.id, {
                repo_id: repoId,
                private: isPrivate,
                commit_message: commitMessage,
                repo_description: repoDescription,
            });

            uploadSuccess = true;
            uploadedUrl = response.url;

            if (onSuccess) {
                onSuccess(response.url);
            }

            // Auto-close after 3 seconds
            setTimeout(() => {
                handleClose();
            }, 3000);
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to upload model";
        } finally {
            uploading = false;
        }
    }

    function handleClose() {
        if (uploading) return; // Don't close while uploading
        uploadSuccess = false;
        uploadedUrl = "";
        error = "";
        onClose();
    }

    function handleBackdropClick(e: MouseEvent) {
        if (e.target === e.currentTarget) {
            handleClose();
        }
    }
</script>

{#if isOpen}
    <div
        class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
        onclick={handleBackdropClick}
        onkeydown={(e) => e.key === "Escape" && handleClose()}
        role="dialog"
        aria-modal="true"
        tabindex="-1"
    >
        <div
            class="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
        >
            <!-- Header -->
            <div
                class="px-6 py-4 border-b border-gray-200 flex items-center justify-between"
            >
                <div>
                    <h2 class="text-xl font-bold text-gray-900">
                        Upload to HuggingFace Hub
                    </h2>
                    <p class="text-sm text-gray-500 mt-1">
                        Share your model with the community
                    </p>
                </div>
                {#if !uploading}
                    <button
                        onclick={handleClose}
                        class="text-gray-400 hover:text-gray-600 transition-colors"
                        aria-label="Close"
                    >
                        <svg
                            class="w-6 h-6"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M6 18L18 6M6 6l12 12"
                            />
                        </svg>
                    </button>
                {/if}
            </div>

            <!-- Content -->
            <div class="px-6 py-4 space-y-6">
                {#if uploadSuccess}
                    <!-- Success State -->
                    <div class="text-center py-8">
                        <div class="text-green-500 text-6xl mb-4">✓</div>
                        <h3 class="text-xl font-bold text-gray-900 mb-2">
                            Model Uploaded Successfully!
                        </h3>
                        <p class="text-gray-600 mb-4">
                            Your model is now available on HuggingFace Hub
                        </p>
                        <a
                            href={uploadedUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            class="inline-flex items-center gap-2 text-primary-600 hover:text-primary-700 font-medium"
                        >
                            View on HuggingFace Hub
                            <svg
                                class="w-4 h-4"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                                />
                            </svg>
                        </a>
                    </div>
                {:else}
                    <!-- Model Info -->
                    <div class="bg-gray-50 rounded-lg p-4">
                        <h3 class="text-sm font-semibold text-gray-700 mb-2">
                            Model Details
                        </h3>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-gray-600">Name:</span>
                                <span class="font-medium text-gray-900"
                                    >{model.name}</span
                                >
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">Base Model:</span>
                                <span class="font-medium text-gray-900"
                                    >{model.base_model}</span
                                >
                            </div>
                            {#if model.size_bytes}
                                <div class="flex justify-between">
                                    <span class="text-gray-600">Size:</span>
                                    <span class="font-medium text-gray-900">
                                        {(model.size_bytes / 1024 ** 3).toFixed(
                                            2,
                                        )} GB
                                    </span>
                                </div>
                            {/if}
                        </div>
                    </div>

                    <!-- Form -->
                    <div class="space-y-4">
                        <!-- Repository ID -->
                        <div>
                            <label
                                for="repo-id"
                                class="block text-sm font-medium text-gray-700 mb-1"
                            >
                                Repository ID <span class="text-red-500">*</span
                                >
                            </label>
                            <input
                                id="repo-id"
                                type="text"
                                bind:value={repoId}
                                placeholder="username/model-name"
                                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                disabled={uploading}
                                required
                            />
                            <p class="mt-1 text-xs text-gray-500">
                                Format: your-username/model-name (e.g.,
                                john/my-finetuned-model)
                            </p>
                        </div>

                        <!-- Visibility -->
                        <div class="flex items-center">
                            <input
                                id="is-private"
                                type="checkbox"
                                bind:checked={isPrivate}
                                class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                                disabled={uploading}
                            />
                            <label
                                for="is-private"
                                class="ml-2 block text-sm text-gray-700"
                            >
                                Make repository private
                            </label>
                        </div>

                        <!-- Commit Message -->
                        <div>
                            <label
                                for="commit-message"
                                class="block text-sm font-medium text-gray-700 mb-1"
                            >
                                Commit Message
                            </label>
                            <input
                                id="commit-message"
                                type="text"
                                bind:value={commitMessage}
                                placeholder="Upload model from Model Garden"
                                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                disabled={uploading}
                            />
                        </div>

                        <!-- Repository Description -->
                        <div>
                            <label
                                for="repo-description"
                                class="block text-sm font-medium text-gray-700 mb-1"
                            >
                                Repository Description
                            </label>
                            <textarea
                                id="repo-description"
                                bind:value={repoDescription}
                                rows="3"
                                placeholder="Describe your model..."
                                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                                disabled={uploading}
                            ></textarea>
                        </div>
                    </div>

                    <!-- Warning -->
                    <div
                        class="bg-yellow-50 border border-yellow-200 rounded-lg p-4"
                    >
                        <div class="flex gap-3">
                            <div class="text-yellow-600 text-xl">⚠️</div>
                            <div class="text-sm text-yellow-800">
                                <p class="font-semibold mb-1">
                                    Before uploading:
                                </p>
                                <ul class="list-disc ml-4 space-y-1">
                                    <li>
                                        Make sure you have a HuggingFace account
                                        and your HF_TOKEN is configured
                                    </li>
                                    <li>
                                        Ensure you have the necessary
                                        permissions to create repositories
                                    </li>
                                    <li>
                                        The entire model directory will be
                                        uploaded (this may take some time)
                                    </li>
                                    <li>
                                        Public repositories will be visible to
                                        everyone on HuggingFace Hub
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {#if error}
                        <div
                            class="bg-red-50 border border-red-200 rounded-lg p-4"
                        >
                            <div class="flex gap-3">
                                <div class="text-red-600 text-xl">❌</div>
                                <div class="text-sm text-red-800">
                                    <p class="font-semibold mb-1">
                                        Upload failed:
                                    </p>
                                    <p>{error}</p>
                                </div>
                            </div>
                        </div>
                    {/if}
                {/if}
            </div>

            <!-- Footer -->
            {#if !uploadSuccess}
                <div
                    class="px-6 py-4 border-t border-gray-200 flex justify-end gap-3"
                >
                    <Button
                        variant="secondary"
                        onclick={handleClose}
                        disabled={uploading}
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="primary"
                        onclick={handleUpload}
                        disabled={uploading || !repoId}
                    >
                        {#if uploading}
                            <span class="flex items-center gap-2">
                                <span
                                    class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"
                                ></span>
                                Uploading...
                            </span>
                        {:else}
                            🚀 Upload to Hub
                        {/if}
                    </Button>
                </div>
            {/if}
        </div>
    </div>
{/if}

<style>
    /* Ensure modal is above everything */
    :global(body:has(.fixed.z-50)) {
        overflow: hidden;
    }
</style>
